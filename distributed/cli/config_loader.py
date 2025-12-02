"""Configuration loader for distributed training CLI.

Loads configuration from YAML file and applies device-specific overrides.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


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
        'redis_host': redis.get('host', 'localhost'),
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

    return {
        'batch_size': batch_size,
        'num_simulations': mcts.get('simulations', 100),
        'max_nodes': mcts.get('max_nodes', 400),
        'temperature': mcts.get('temperature', 1.0),
        'max_episode_steps': game.get('max_episode_steps', 500),
        'redis_host': redis.get('host', 'localhost'),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
    }


def get_training_worker_config(
    config: Dict[str, Any],
    device_type: Optional[str] = None,
    batch_size_override: Optional[int] = None
) -> Dict[str, Any]:
    """Extract training worker configuration with device-specific batch size.

    Args:
        config: Full configuration dictionary.
        device_type: Device type. Auto-detected if None.
        batch_size_override: Override batch size (from CLI --batch-size).

    Returns:
        Training worker config dict.
    """
    if device_type is None:
        device_type = detect_device_type()

    training = config.get('training', {})
    redis = config.get('redis', {})
    device_cfg = get_device_config(config, device_type)

    # Batch size priority: CLI override > device config > global default
    if batch_size_override is not None:
        batch_size = batch_size_override
    else:
        batch_size = device_cfg.get('train_batch_size', training.get('batch_size', 128))

    return {
        'train_batch_size': batch_size,
        'learning_rate': training.get('learning_rate', 3e-4),
        'l2_reg_lambda': training.get('l2_reg_lambda', 1e-4),
        'checkpoint_interval': training.get('checkpoint_interval', 1000),
        'min_buffer_size': 1000,  # Could add to config
        'checkpoint_dir': config.get('checkpoint_dir', './checkpoints'),
        'redis_host': redis.get('host', 'localhost'),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
        'games_per_training_batch': training.get('games_per_batch', 10),
        'steps_per_game': training.get('steps_per_game', 10),
        'surprise_weight': training.get('surprise_weight', 0.5),
    }


def get_eval_worker_config(
    config: Dict[str, Any],
    device_type: Optional[str] = None,
    batch_size_override: Optional[int] = None
) -> Dict[str, Any]:
    """Extract evaluation worker configuration.

    Args:
        config: Full configuration dictionary.
        device_type: Device type. Auto-detected if None.
        batch_size_override: Override batch size (from CLI --batch-size).

    Returns:
        Eval worker config dict.
    """
    if device_type is None:
        device_type = detect_device_type()

    mcts = config.get('mcts', {})
    redis = config.get('redis', {})
    device_cfg = get_device_config(config, device_type)

    # Batch size priority: CLI override > device config > default
    if batch_size_override is not None:
        batch_size = batch_size_override
    else:
        batch_size = device_cfg.get('eval_batch_size', 16)

    return {
        'batch_size': batch_size,
        'num_simulations': mcts.get('simulations', 100),
        'max_nodes': mcts.get('max_nodes', 400),
        'redis_host': redis.get('host', 'localhost'),
        'redis_port': redis.get('port', 6379),
        'redis_password': redis.get('password'),
    }


def get_ray_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Ray cluster configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Ray config dict with head_ip, ports, etc.
    """
    ray = config.get('ray', {})
    return {
        'head_ip': ray.get('head_ip', '100.105.50.111'),
        'head_ip_local': ray.get('head_ip_local', '192.168.20.40'),
        'gcs_port': ray.get('gcs_port', 6380),
        'client_port': ray.get('client_port', 10001),
        'dashboard_port': ray.get('dashboard_port', 8265),
    }


def print_config_summary(config: Dict[str, Any], device_type: str):
    """Print a summary of the loaded configuration."""
    mcts = config.get('mcts', {})
    device_cfg = get_device_config(config, device_type)

    print(f"\n=== Configuration Summary ===")
    print(f"Device type: {device_type}")
    print(f"MCTS simulations: {mcts.get('simulations', 100)} (global)")
    print(f"MCTS max nodes: {mcts.get('max_nodes', 400)} (global)")
    print(f"Game batch size: {device_cfg.get('game_batch_size', 'default')}")
    print(f"Train batch size: {device_cfg.get('train_batch_size', 'default')}")
    print(f"Eval batch size: {device_cfg.get('eval_batch_size', 'default')}")
    print()
