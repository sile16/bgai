"""Configuration management for distributed training.

This module provides dataclasses for configuration and utilities for
loading/saving YAML configuration files.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class MCTSConfig:
    """MCTS evaluator configuration."""
    collect_simulations: Optional[int] = None  # MCTS iterations for game collection
    eval_simulations: Optional[int] = None     # MCTS iterations for evaluation
    max_nodes: Optional[int] = None
    warm_tree_simulations: Optional[int] = None  # MCTS sims for warm tree (0 = disabled)
    persist_tree: Optional[bool] = None
    temperature_start: Optional[float] = None
    temperature_end: Optional[float] = None
    temperature_epochs: Optional[int] = None    # Epochs to decay from start to end
    discount: Optional[float] = None  # -1 for two-player zero-sum games


@dataclass
class TrainingConfig:
    """Neural network training configuration."""
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    l2_reg_lambda: Optional[float] = None
    games_per_epoch: Optional[int] = None           # New games before training epoch
    steps_per_epoch: Optional[int] = None           # Training steps per epoch
    surprise_weight: Optional[float] = None         # Blend of uniform vs surprise-weighted sampling
    checkpoint_epoch_interval: Optional[int] = None # Epochs between checkpoints
    max_checkpoints: Optional[int] = None
    bearoff_enabled: Optional[bool] = None
    bearoff_value_weight: Optional[float] = None
    lookup_enabled: Optional[bool] = None
    lookup_learning_weight: Optional[float] = None


@dataclass
class GameConfig:
    """Game/self-play configuration."""
    batch_size: Optional[int] = None  # Parallel games per worker
    max_episode_steps: Optional[int] = None
    short_game: Optional[bool] = None         # Start from mid-game position
    simple_doubles: Optional[bool] = None    # Limit randomness with simple doubles


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: Optional[str] = None
    port: Optional[int] = None
    db: Optional[int] = None
    password: Optional[str] = None
    buffer_capacity: Optional[int] = None
    episode_capacity: Optional[int] = None


@dataclass
class CoordinatorConfig:
    """Coordinator/head node configuration."""
    heartbeat_timeout: Optional[float] = None
    heartbeat_interval: Optional[float] = None
    weight_push_interval: Optional[int] = None  # Push weights every N training steps
    ray_port: Optional[int] = None
    dashboard_port: Optional[int] = None


@dataclass
class WorkerConfig:
    """Worker node configuration."""
    worker_type: Optional[str] = None  # 'game', 'training', 'evaluation'
    prefer_gpu: Optional[bool] = None
    auto_config: Optional[bool] = None  # Auto-configure based on device


@dataclass
class NetworkConfig:
    """Neural network architecture configuration."""
    hidden_dim: Optional[int] = None
    num_blocks: Optional[int] = None
    num_actions: Optional[int] = None  # Backgammon action space


@dataclass
class DistributedConfig:
    """Complete distributed training configuration."""
    # Sub-configurations
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # Paths
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None

    # Device-specific overrides (applied based on detected device)
    cuda_overrides: Dict[str, Any] = field(default_factory=dict)
    metal_overrides: Dict[str, Any] = field(default_factory=dict)
    cpu_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def apply_device_overrides(self, device_type: str) -> 'DistributedConfig':
        """Apply device-specific overrides and return new config.

        Args:
            device_type: One of 'cuda', 'metal', 'cpu'

        Returns:
            New DistributedConfig with overrides applied.
        """
        import copy
        config = copy.deepcopy(self)

        if device_type == 'cuda':
            overrides = self.cuda_overrides
        elif device_type == 'metal':
            overrides = self.metal_overrides
        else:
            overrides = self.cpu_overrides

        # Apply overrides
        for section, values in overrides.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)

        return config


# =============================================================================
# YAML Loading/Saving
# =============================================================================

def load_config(config_path: Union[str, Path]) -> DistributedConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        DistributedConfig instance.

    Example:
        >>> config = load_config('configs/distributed.yaml')
        >>> print(config.mcts.simulations)
        100
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return _dict_to_config(data)


def save_config(config: DistributedConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: DistributedConfig instance.
        config_path: Path to save YAML file.

    Example:
        >>> save_config(config, 'configs/my_config.yaml')
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(config_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _dict_to_config(data: Dict[str, Any]) -> DistributedConfig:
    """Convert dictionary to DistributedConfig.

    Args:
        data: Dictionary with configuration values.

    Returns:
        DistributedConfig instance.
    """
    config = DistributedConfig()

    # Update sub-configurations
    if 'mcts' in data:
        config.mcts = MCTSConfig(**data['mcts'])
    if 'training' in data:
        config.training = TrainingConfig(**data['training'])
    if 'game' in data:
        config.game = GameConfig(**data['game'])
    if 'redis' in data:
        config.redis = RedisConfig(**data['redis'])
    if 'coordinator' in data:
        config.coordinator = CoordinatorConfig(**data['coordinator'])
    if 'worker' in data:
        config.worker = WorkerConfig(**data['worker'])
    if 'network' in data:
        config.network = NetworkConfig(**data['network'])

    # Update top-level settings
    if 'checkpoint_dir' in data:
        config.checkpoint_dir = data['checkpoint_dir']
    if 'log_dir' in data:
        config.log_dir = data['log_dir']

    # Update device overrides
    if 'cuda_overrides' in data:
        config.cuda_overrides = data['cuda_overrides']
    if 'metal_overrides' in data:
        config.metal_overrides = data['metal_overrides']
    if 'cpu_overrides' in data:
        config.cpu_overrides = data['cpu_overrides']

    return config


# =============================================================================
# Default Configuration
# =============================================================================

def get_default_config() -> DistributedConfig:
    """Get default distributed training configuration.

    Returns:
        DistributedConfig with default values.
    """
    return DistributedConfig()


def get_config_for_device(device_info: Optional[Any] = None) -> DistributedConfig:
    """Get configuration optimized for the current device.

    Args:
        device_info: Optional DeviceInfo from device.py. If None, will auto-detect.

    Returns:
        DistributedConfig with device-specific optimizations.

    Example:
        >>> config = get_config_for_device()
        >>> # config is now optimized for your GPU/CPU
    """
    if device_info is None:
        from .device import detect_device
        device_info = detect_device()

    config = DistributedConfig()

    if device_info.is_cuda:
        config = config.apply_device_overrides('cuda')
    elif device_info.is_metal:
        config = config.apply_device_overrides('metal')
    else:
        config = config.apply_device_overrides('cpu')

    return config


def create_default_config_file(config_path: Union[str, Path] = "configs/distributed.yaml") -> None:
    """Create a default configuration file.

    Args:
        config_path: Path to save the default configuration.

    Example:
        >>> create_default_config_file()
        >>> # Creates configs/distributed.yaml with default values
    """
    config = get_default_config()
    save_config(config, config_path)
    print(f"Created default config at: {config_path}")


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: DistributedConfig) -> list:
    """Validate configuration and return list of issues.

    Args:
        config: DistributedConfig to validate.

    Returns:
        List of validation error strings. Empty if valid.

    Example:
        >>> issues = validate_config(config)
        >>> if issues:
        ...     print("Config issues:", issues)
    """
    issues = []

    # MCTS validation
    if config.mcts.collect_simulations < 1:
        issues.append("mcts.collect_simulations must be >= 1")
    if config.mcts.max_nodes < config.mcts.collect_simulations:
        issues.append("mcts.max_nodes should be >= mcts.collect_simulations")
    if not 0 <= config.mcts.temperature_start <= 10:
        issues.append("mcts.temperature_start should be in [0, 10]")
    if not 0 <= config.mcts.temperature_end <= 10:
        issues.append("mcts.temperature_end should be in [0, 10]")

    # Training validation
    if config.training.batch_size < 1:
        issues.append("training.batch_size must be >= 1")
    if config.training.learning_rate <= 0:
        issues.append("training.learning_rate must be > 0")

    # Game validation
    if config.game.batch_size < 1:
        issues.append("game.batch_size must be >= 1")
    if config.game.max_episode_steps < 1:
        issues.append("game.max_episode_steps must be >= 1")

    # Redis validation
    if config.redis.port < 1 or config.redis.port > 65535:
        issues.append("redis.port must be in [1, 65535]")
    if config.redis.buffer_capacity < 1000:
        issues.append("redis.buffer_capacity should be >= 1000")

    # Coordinator validation
    if config.coordinator.heartbeat_timeout <= 0:
        issues.append("coordinator.heartbeat_timeout must be > 0")
    if config.coordinator.heartbeat_interval <= 0:
        issues.append("coordinator.heartbeat_interval must be > 0")
    if config.coordinator.heartbeat_interval >= config.coordinator.heartbeat_timeout:
        issues.append("coordinator.heartbeat_interval should be < heartbeat_timeout")

    return issues


# =============================================================================
# CLI Config Helpers
# =============================================================================

def merge_cli_args(config: DistributedConfig, cli_args: Dict[str, Any]) -> DistributedConfig:
    """Merge CLI arguments into configuration.

    CLI arguments override config file values.

    Args:
        config: Base DistributedConfig.
        cli_args: Dictionary of CLI arguments.

    Returns:
        Updated DistributedConfig.

    Example:
        >>> config = load_config('config.yaml')
        >>> config = merge_cli_args(config, {'redis_host': '192.168.1.100'})
    """
    import copy
    config = copy.deepcopy(config)

    # Map CLI arg names to config paths
    cli_mapping = {
        'redis_host': ('redis', 'host'),
        'redis_port': ('redis', 'port'),
        'mcts_simulations': ('mcts', 'collect_simulations'),
        'mcts_max_nodes': ('mcts', 'max_nodes'),
        'train_batch_size': ('training', 'batch_size'),
        'learning_rate': ('training', 'learning_rate'),
        'games_per_epoch': ('training', 'games_per_epoch'),
        'checkpoint_epoch_interval': ('training', 'checkpoint_epoch_interval'),
        'game_batch_size': ('game', 'batch_size'),
        'checkpoint_dir': ('checkpoint_dir', None),
        'worker_type': ('worker', 'worker_type'),
    }

    for cli_name, config_path in cli_mapping.items():
        if cli_name in cli_args and cli_args[cli_name] is not None:
            if config_path[1] is None:
                # Top-level attribute
                setattr(config, config_path[0], cli_args[cli_name])
            else:
                # Nested attribute
                section = getattr(config, config_path[0])
                setattr(section, config_path[1], cli_args[cli_name])

    return config
