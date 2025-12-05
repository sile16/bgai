"""Configuration loader for distributed training CLI.

Loads configuration from YAML file and applies device-specific overrides.
"""

import os
import socket
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def is_host_reachable(host: str, port: int = 6379, timeout: float = 1.0) -> bool:
    """Check if a host:port is reachable.

    Args:
        host: Hostname or IP address.
        port: Port number to check.
        timeout: Connection timeout in seconds.

    Returns:
        True if reachable, False otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (socket.error, socket.timeout):
        return False


def detect_redis_host(config: Dict[str, Any]) -> str:
    """Auto-detect the reachable Redis host.

    Tries head node IP first, then head_local fallback, then to redis.host config.

    Args:
        config: Full configuration dictionary.

    Returns:
        Reachable Redis host IP/hostname.
    """
    head = config.get('head', {})
    redis = config.get('redis', {})
    redis_port = redis.get('port', 6379)

    # Get candidate IPs
    head_ip = head.get('host')
    head_local_ip = head.get('host_local')
    config_host = redis.get('host', 'localhost')

    # Try head node IP first
    if head_ip and is_host_reachable(head_ip, redis_port):
        return head_ip

    # Try local/LAN IP next
    if head_local_ip and is_host_reachable(head_local_ip, redis_port):
        return head_local_ip

    # Fall back to config host
    return config_host


def find_config_file() -> Optional[Path]:
    """Find the default config file.

    Searches in order:
    1. ./configs/distributed.yaml (current directory)
    2. <project_root>/configs/distributed.yaml
    """
    # Check current directory
    local_config = Path("configs/distributed.yaml")
    if local_config.exists():
        return local_config

    # Check project root (relative to this file)
    project_root = Path(__file__).parent.parent.parent
    project_config = project_root / "configs" / "distributed.yaml"
    if project_config.exists():
        return project_config

    return None


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches for default.

    Returns:
        Configuration dictionary.
    """
    if config_path:
        path = Path(config_path)
    else:
        path = find_config_file()

    if path is None or not path.exists():
        print(f"Warning: Config file not found, using defaults")
        return {}

    print(f"Loading config from: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def detect_device_type() -> str:
    """Detect the current device type.

    Returns:
        One of: 'cuda', 'metal', 'tpu', 'cpu'
    """
    try:
        import jax
        devices = jax.devices()
        if devices:
            platform = devices[0].platform
            if platform == 'gpu':
                # Check if NVIDIA CUDA
                device_kind = str(devices[0].device_kind).lower()
                if 'nvidia' in device_kind or 'cuda' in device_kind:
                    return 'cuda'
                return 'cuda'  # Assume CUDA for GPU
            elif platform == 'tpu':
                return 'tpu'
            elif platform == 'cpu':
                # Check if Metal is available (macOS)
                import platform as plat
                if plat.system() == 'Darwin':
                    return 'metal'
                return 'cpu'
    except Exception:
        pass

    # Fallback detection
    import platform as plat
    if plat.system() == 'Darwin':
        return 'metal'

    # Check for CUDA
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            return 'cuda'
    except Exception:
        pass

    return 'cpu'


def get_device_config(config: Dict[str, Any], device_type: Optional[str] = None) -> Dict[str, Any]:
    """Get device-specific configuration values.

    Args:
        config: Full configuration dictionary.
        device_type: Device type ('cuda', 'metal', 'tpu', 'cpu'). Auto-detected if None.

    Returns:
        Device-specific config values.
    """
    if device_type is None:
        device_type = detect_device_type()

    device_configs = config.get('device_configs', {})
    return device_configs.get(device_type, device_configs.get('cpu', {}))


def get_coordinator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract coordinator configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Coordinator config dict ready for use.
    """
    mcts = config.get('mcts', {})
    training = config.get('training', {})
    redis = config.get('redis', {})
    coord = config.get('coordinator', {})

    return {
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        'heartbeat_timeout': coord.get('heartbeat_timeout', 30.0),
        'heartbeat_interval': coord.get('heartbeat_interval', 10.0),
        'mcts_simulations': mcts.get('simulations', 100),
        'mcts_max_nodes': mcts.get('max_nodes', 400),
        'train_batch_size': training.get('batch_size', 128),
        'game_batch_size': config.get('game', {}).get('batch_size', 16),
        'learning_rate': training.get('learning_rate', 3e-4),
        # Store full config for device-specific distribution
        'device_configs': config.get('device_configs', {}),
    }


def get_game_worker_config(
    config: Dict[str, Any],
    device_type: Optional[str] = None,
    batch_size_override: Optional[int] = None
) -> Dict[str, Any]:
    """Extract game worker configuration with device-specific batch size.

    Args:
        config: Full configuration dictionary.
        device_type: Device type. Auto-detected if None.
        batch_size_override: Override batch size (from CLI --batch-size).

    Returns:
        Game worker config dict.
    """
    if device_type is None:
        device_type = detect_device_type()

    mcts = config.get('mcts', {})
    game = config.get('game', {})
    redis = config.get('redis', {})
    device_cfg = get_device_config(config, device_type)

    # Batch size priority: CLI override > device config > global default
    if batch_size_override is not None:
        batch_size = batch_size_override
    else:
        batch_size = device_cfg.get('game_batch_size', game.get('batch_size', 16))

    # Temperature schedule: start high for exploration, decay to low for exploitation
    # If temperature_start/end are defined, use schedule; otherwise use static temperature
    temp_start = game.get('temperature_start')
    temp_end = game.get('temperature_end')
    static_temp = mcts.get('temperature', 1.0)

    return {
        'batch_size': batch_size,
        'num_simulations': mcts.get('simulations', 100),
        'max_nodes': mcts.get('max_nodes', 400),
        # Static temperature (used if no schedule defined)
        'temperature': static_temp,
        # Temperature schedule (None means use static)
        'temperature_start': temp_start,
        'temperature_end': temp_end,
        'max_episode_steps': game.get('max_episode_steps', 500),
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
    }


def get_training_worker_config(
    config: Dict[str, Any],
    device_type: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    head_ip: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract training worker configuration with device-specific batch size.

    Args:
        config: Full configuration dictionary.
        device_type: Device type. Auto-detected if None.
        batch_size_override: Override batch size (from CLI --batch-size).
        head_ip: Head node IP for MLflow/services. Uses auto-detection if None.

    Returns:
        Training worker config dict.
    """
    if device_type is None:
        device_type = detect_device_type()

    training = config.get('training', {})
    redis = config.get('redis', {})
    mlflow = config.get('mlflow', {})
    device_cfg = get_device_config(config, device_type)

    # Batch size priority: CLI override > device config > global default
    if batch_size_override is not None:
        batch_size = batch_size_override
    else:
        batch_size = device_cfg.get('train_batch_size', training.get('batch_size', 128))

    # Build MLFlow tracking URI - use head_ip if provided, else auto-detect
    mlflow_uri = mlflow.get('tracking_uri')
    if mlflow_uri:
        # Determine the target host for MLflow
        if head_ip:
            # Use explicitly provided head IP
            target_host = head_ip
        else:
            # Auto-detect by finding reachable head node (same as Redis detection)
            target_host = detect_redis_host(config)

        # Replace localhost, head IPs, etc. with target host
        head = config.get('head', {})
        config_head_ip = head.get('host', '')
        config_head_local = head.get('host_local', '')

        if 'localhost' in mlflow_uri or '127.0.0.1' in mlflow_uri:
            mlflow_uri = mlflow_uri.replace('localhost', target_host).replace('127.0.0.1', target_host)
        elif config_head_ip and config_head_ip in mlflow_uri:
            mlflow_uri = mlflow_uri.replace(config_head_ip, target_host)
        elif config_head_local and config_head_local in mlflow_uri:
            mlflow_uri = mlflow_uri.replace(config_head_local, target_host)

    return {
        'train_batch_size': batch_size,
        'learning_rate': training.get('learning_rate', 3e-4),
        'l2_reg_lambda': training.get('l2_reg_lambda', 1e-4),
        'checkpoint_interval': training.get('checkpoint_interval', 1000),
        'min_buffer_size': 1000,  # Could add to config
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        'games_per_training_batch': training.get('games_per_batch', 10),
        'steps_per_game': training.get('steps_per_game', 10),
        'surprise_weight': training.get('surprise_weight', 0.5),
        # Warm tree configuration (pre-computed MCTS tree from initial position)
        'warm_tree_simulations': training.get('warm_tree_simulations', 0),
        'warm_tree_max_nodes': training.get('warm_tree_max_nodes', 10000),
        # MLFlow tracking
        'mlflow_tracking_uri': mlflow_uri,
        'mlflow_experiment_name': mlflow.get('experiment_name', 'bgai-training'),
    }


def get_eval_worker_config(
    config: Dict[str, Any],
    device_type: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    head_ip: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract evaluation worker configuration.

    Args:
        config: Full configuration dictionary.
        device_type: Device type. Auto-detected if None.
        batch_size_override: Override batch size (from CLI --batch-size).
        head_ip: Head node IP for MLflow/services. Uses auto-detection if None.

    Returns:
        Eval worker config dict.
    """
    if device_type is None:
        device_type = detect_device_type()

    mcts = config.get('mcts', {})
    redis = config.get('redis', {})
    mlflow = config.get('mlflow', {})
    device_cfg = get_device_config(config, device_type)

    # Batch size priority: CLI override > device config > default
    if batch_size_override is not None:
        batch_size = batch_size_override
    else:
        batch_size = device_cfg.get('eval_batch_size', 16)

    # Build MLFlow tracking URI - use head_ip if provided, else auto-detect
    mlflow_uri = mlflow.get('tracking_uri')
    if mlflow_uri:
        # Determine the target host for MLflow
        if head_ip:
            # Use explicitly provided head IP
            target_host = head_ip
        else:
            # Auto-detect by finding reachable head node (same as Redis detection)
            target_host = detect_redis_host(config)

        # Replace localhost, head IPs, etc. with target host
        head = config.get('head', {})
        config_head_ip = head.get('host', '')
        config_head_local = head.get('host_local', '')

        if 'localhost' in mlflow_uri or '127.0.0.1' in mlflow_uri:
            mlflow_uri = mlflow_uri.replace('localhost', target_host).replace('127.0.0.1', target_host)
        elif config_head_ip and config_head_ip in mlflow_uri:
            mlflow_uri = mlflow_uri.replace(config_head_ip, target_host)
        elif config_head_local and config_head_local in mlflow_uri:
            mlflow_uri = mlflow_uri.replace(config_head_local, target_host)

    return {
        'batch_size': batch_size,
        'num_simulations': mcts.get('simulations', 100),
        'max_nodes': mcts.get('max_nodes', 400),
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        # MLFlow tracking (shared with training worker)
        'mlflow_tracking_uri': mlflow_uri,
        'mlflow_experiment_name': mlflow.get('experiment_name', 'bgai-training'),
    }


def print_config_summary(config: Dict[str, Any], device_type: str):
    """Print a summary of the loaded configuration."""
    mcts = config.get('mcts', {})
    device_cfg = get_device_config(config, device_type)
    redis_host = detect_redis_host(config)
    redis_port = config.get('redis', {}).get('port', 6379)

    print(f"\n=== Configuration Summary ===")
    print(f"Device type: {device_type}")
    print(f"Redis host: {redis_host}:{redis_port} (auto-detected)")
    print(f"MCTS simulations: {mcts.get('simulations', 100)} (global)")
    print(f"MCTS max nodes: {mcts.get('max_nodes', 400)} (global)")
    print(f"Game batch size: {device_cfg.get('game_batch_size', 'default')}")
    print(f"Train batch size: {device_cfg.get('train_batch_size', 'default')}")
    print(f"Eval batch size: {device_cfg.get('eval_batch_size', 'default')}")
    print()
