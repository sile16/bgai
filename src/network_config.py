# network_config.py
from dataclasses import dataclass
from typing import Tuple, List, Union, Dict, Callable, Optional

@dataclass
class NetworkConfig:
    """Configuration for network architecture parameters."""
    input_channels: int = 30
    input_size: int = 24
    conv_channels: int = 128
    policy_channels: int = 64
    value_channels: int = 32
    hidden_size: int = 256
    num_blocks: int = 3
    kernel_size: int = 3
    action_space_size: int = 7128
    
@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 1000
    batches_per_epoch: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_workers: int = 4
    buffer_size: int = 10000
    min_games_to_start: int = 100
    eval_interval: int = 10
    eval_games: int = 50
    eval_positions: int = 100
    save_interval: int = 10
    patience: int = 20
    save_dir: str = "checkpoints"
    # Learning rate schedule: either None, dict mapping epoch -> lr, or callable
    lr_schedule: Optional[Union[Dict[int, float], Callable[[int], float]]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lr_schedule is None:
            # Default schedule: reduce learning rate by factor of 10 at epochs 500 and 750
            self.lr_schedule = {
                0: self.learning_rate,
                500: self.learning_rate * 0.1,
                750: self.learning_rate * 0.01
            }
