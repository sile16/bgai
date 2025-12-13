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
        'mcts_simulations': mcts.get('collect_simulations', mcts.get('simulations', 100)),
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

    # Temperature schedule: now in mcts section
    # start high for exploration, decay to low for exploitation
    temp_start = mcts.get('temperature_start')
    temp_end = mcts.get('temperature_end')

    # Network config for model architecture
    network = config.get('network', {})

    return {
        'batch_size': batch_size,
        # collect_simulations is used for game generation (was just 'simulations')
        'num_simulations': mcts.get('collect_simulations', mcts.get('simulations', 100)),
        'max_nodes': mcts.get('max_nodes', 400),
        # Temperature schedule (None means use static default of 1.0)
        'temperature': 1.0,  # Default if no schedule
        'temperature_start': temp_start,
        'temperature_end': temp_end,
        'temperature_epochs': mcts.get('temperature_epochs', 50),
        # Game settings
        'max_episode_steps': game.get('max_episode_steps', 500),
        'short_game': game.get('short_game', True),
        'simple_doubles': game.get('simple_doubles', False),
        # Redis connection
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        # Network architecture
        'network_hidden_dim': network.get('hidden_dim', 256),
        'network_num_blocks': network.get('num_blocks', 6),
        'network_num_actions': network.get('num_actions', 156),
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

    # Determine the target host for MLflow
    if head_ip:
        target_host = head_ip
    else:
        target_host = detect_redis_host(config)

    if mlflow_uri:
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
    else:
        # Build URI from head.host and mlflow.port
        mlflow_port = mlflow.get('port', 5000)
        mlflow_uri = f"http://{target_host}:{mlflow_port}"

    mcts = config.get('mcts', {})

    # Network config for model architecture
    network = config.get('network', {})
    games_per_epoch = training.get('games_per_epoch', training.get('games_per_batch', 10))
    steps_per_epoch = training.get('steps_per_epoch')

    worker_config = {
        'train_batch_size': batch_size,
        'learning_rate': training.get('learning_rate', 3e-4),
        'l2_reg_lambda': training.get('l2_reg_lambda', 1e-4),
        # checkpoint_epoch_interval (new) with fallback to checkpoint_interval (old)
        'checkpoint_epoch_interval': training.get('checkpoint_epoch_interval', training.get('checkpoint_interval', 5)),
        'max_checkpoints': training.get('max_checkpoints', 5),
        'min_buffer_size': 1000,  # Could add to config
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        # games_per_epoch (new) with fallback to games_per_batch (old)
        'games_per_epoch': games_per_epoch,
        # Training/collection scheduling
        'pause_collection_during_training': training.get('pause_collection_during_training', False),
        'surprise_weight': training.get('surprise_weight', 0.5),
        # CPU-side pipeline tuning (deserialization/stacking)
        'decode_threads': training.get('decode_threads', 0),
        # Increase GPU work per CPU decode by reusing the same minibatch for N updates.
        'batch_reuse_steps': training.get('batch_reuse_steps', 1),
        # Bearoff/endgame settings
        'bearoff_enabled': training.get('bearoff_enabled', False),
        'bearoff_table_path': training.get('bearoff_table_path'),
        'bearoff_value_weight': training.get('bearoff_value_weight', 2.0),
        'lookup_enabled': training.get('lookup_enabled', False),
        'lookup_learning_weight': training.get('lookup_learning_weight', 1.5),
        # Warm tree configuration (now in mcts section)
        'warm_tree_simulations': mcts.get('warm_tree_simulations', 0),
        'warm_tree_max_nodes': mcts.get('warm_tree_max_nodes', 10000),
        # Temperature schedule
        'temperature_start': mcts.get('temperature_start', 0.8),
        'temperature_end': mcts.get('temperature_end', 0.2),
        'temperature_epochs': mcts.get('temperature_epochs', 50),
        # MLFlow tracking
        'mlflow_tracking_uri': mlflow_uri,
        'mlflow_experiment_name': mlflow.get('experiment_name', 'bgai-training'),
        # Network architecture
        'network_hidden_dim': network.get('hidden_dim', 256),
        'network_num_blocks': network.get('num_blocks', 6),
        'network_num_actions': network.get('num_actions', 156),
        # Pass full config sections for MLflow param logging
        'mcts': mcts,
        'game': config.get('game', {}),
        'network': network,
        'redis': redis,
        'gnubg': config.get('gnubg', {}),
    }

    # Training cadence (optional to keep configs clean when unset)
    if steps_per_epoch is not None:
        worker_config['steps_per_epoch'] = steps_per_epoch

    # Optional scheduling knobs
    if training.get('steps_per_game') is not None:
        worker_config['steps_per_game'] = training.get('steps_per_game')
    if training.get('min_backlog_steps') is not None:
        worker_config['min_backlog_steps'] = training.get('min_backlog_steps')
    if training.get('max_train_steps_per_batch') is not None:
        worker_config['max_train_steps_per_batch'] = training.get('max_train_steps_per_batch')

    return worker_config


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

    # Determine the target host for MLflow
    if head_ip:
        target_host = head_ip
    else:
        target_host = detect_redis_host(config)

    if mlflow_uri:
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
    else:
        # Build URI from head.host and mlflow.port
        mlflow_port = mlflow.get('port', 5000)
        mlflow_uri = f"http://{target_host}:{mlflow_port}"

    # Network config for model architecture
    network = config.get('network', {})

    return {
        'batch_size': batch_size,
        # eval_simulations is used for evaluation (can be different from collect)
        'num_simulations': mcts.get('eval_simulations', mcts.get('collect_simulations', mcts.get('simulations', 100))),
        'max_nodes': mcts.get('max_nodes', 400),
        'redis_host': detect_redis_host(config),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        # MLFlow tracking (shared with training worker)
        'mlflow_tracking_uri': mlflow_uri,
        'mlflow_experiment_name': mlflow.get('experiment_name', 'bgai-training'),
        # Network architecture
        'network_hidden_dim': network.get('hidden_dim', 256),
        'network_num_blocks': network.get('num_blocks', 6),
        'network_num_actions': network.get('num_actions', 156),
    }


def print_config_summary(config: Dict[str, Any], device_type: str):
    """Print a summary of the loaded configuration."""
    mcts = config.get('mcts', {})
    game = config.get('game', {})
    device_cfg = get_device_config(config, device_type)
    redis_host = detect_redis_host(config)
    redis_port = config.get('redis', {}).get('port', 6379)

    collect_sims = mcts.get('collect_simulations', mcts.get('simulations', 100))
    eval_sims = mcts.get('eval_simulations', collect_sims)

    print(f"\n=== Configuration Summary ===")
    print(f"Device type: {device_type}")
    print(f"Redis host: {redis_host}:{redis_port} (auto-detected)")
    print(f"MCTS collect sims: {collect_sims}, eval sims: {eval_sims}")
    print(f"MCTS max nodes: {mcts.get('max_nodes', 400)}")
    print(f"Temperature: {mcts.get('temperature_start', 0.8)} -> {mcts.get('temperature_end', 0.2)}")
    print(f"Game: short_game={game.get('short_game', True)}, simple_doubles={game.get('simple_doubles', False)}")
    print(f"Batch sizes: game={device_cfg.get('game_batch_size', 'default')}, "
          f"train={device_cfg.get('train_batch_size', 'default')}, "
          f"eval={device_cfg.get('eval_batch_size', 'default')}")
    print()
