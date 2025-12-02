"""Device detection and configuration for distributed training.

This module provides automatic detection of JAX devices (CUDA, Metal, CPU)
and returns appropriate configurations for each device type.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import platform


@dataclass
class DeviceInfo:
    """Information about the detected JAX device."""
    platform: str           # 'gpu', 'cpu', 'METAL'
    device_kind: str        # e.g., 'Tesla V100', 'Apple M1', 'cpu'
    device_count: int       # Number of devices
    is_gpu: bool            # True if GPU (CUDA or Metal)
    is_cuda: bool           # True if NVIDIA CUDA
    is_metal: bool          # True if Apple Metal
    is_cpu: bool            # True if CPU only
    jax_version: str        # JAX version string
    system_platform: str    # 'Darwin', 'Linux', 'Windows'

    def __str__(self) -> str:
        device_type = "CUDA" if self.is_cuda else "Metal" if self.is_metal else "CPU"
        return f"{device_type}: {self.device_kind} (x{self.device_count})"


def detect_device() -> DeviceInfo:
    """Detect available JAX device (CUDA, Metal, or CPU).

    Returns:
        DeviceInfo: Information about the detected device.

    Example:
        >>> info = detect_device()
        >>> print(info)
        Metal: Apple M1 (x1)
        >>> print(info.is_metal)
        True
    """
    import os

    # If CUDA_VISIBLE_DEVICES is empty (Ray's way of saying no GPU),
    # force JAX to use CPU to avoid CUDA initialization errors
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible == '':
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax

    devices = jax.devices()
    device = devices[0]

    # Get device platform and kind
    device_platform = device.platform
    device_kind = getattr(device, 'device_kind', str(device))

    # Determine device type
    # Note: Metal shows up as platform='METAL' on Apple Silicon
    is_metal = device_platform.upper() == 'METAL'
    is_cuda = device_platform == 'gpu' and not is_metal
    is_cpu = device_platform == 'cpu'
    is_gpu = is_cuda or is_metal

    return DeviceInfo(
        platform=device_platform,
        device_kind=device_kind,
        device_count=len(devices),
        is_gpu=is_gpu,
        is_cuda=is_cuda,
        is_metal=is_metal,
        is_cpu=is_cpu,
        jax_version=jax.__version__,
        system_platform=platform.system(),
    )


@dataclass
class DeviceConfig:
    """Configuration parameters optimized for a specific device type."""
    mcts_simulations: int
    mcts_max_nodes: int
    game_batch_size: int
    train_batch_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mcts_simulations': self.mcts_simulations,
            'mcts_max_nodes': self.mcts_max_nodes,
            'game_batch_size': self.game_batch_size,
            'train_batch_size': self.train_batch_size,
        }


# Pre-defined configurations for different device types
CUDA_CONFIG = DeviceConfig(
    mcts_simulations=200,
    mcts_max_nodes=800,
    game_batch_size=64,
    train_batch_size=256,
)

METAL_CONFIG = DeviceConfig(
    mcts_simulations=100,
    mcts_max_nodes=400,
    game_batch_size=16,
    train_batch_size=128,
)

CPU_CONFIG = DeviceConfig(
    mcts_simulations=50,
    mcts_max_nodes=200,
    game_batch_size=4,
    train_batch_size=64,
)


def get_device_config(device_info: Optional[DeviceInfo] = None) -> DeviceConfig:
    """Get recommended configuration based on detected device.

    Args:
        device_info: Optional DeviceInfo. If None, will auto-detect.

    Returns:
        DeviceConfig: Configuration optimized for the device.

    Example:
        >>> config = get_device_config()
        >>> print(config.mcts_simulations)
        100  # On Metal
    """
    if device_info is None:
        device_info = detect_device()

    if device_info.is_cuda:
        return CUDA_CONFIG
    elif device_info.is_metal:
        return METAL_CONFIG
    else:
        return CPU_CONFIG


def print_device_info() -> DeviceInfo:
    """Print device information and return DeviceInfo.

    Useful for debugging and verification.

    Returns:
        DeviceInfo: The detected device information.
    """
    info = detect_device()
    config = get_device_config(info)

    print("=" * 50)
    print("JAX Device Information")
    print("=" * 50)
    print(f"Platform:      {info.platform}")
    print(f"Device Kind:   {info.device_kind}")
    print(f"Device Count:  {info.device_count}")
    print(f"JAX Version:   {info.jax_version}")
    print(f"System:        {info.system_platform}")
    print()
    print(f"Is GPU:        {info.is_gpu}")
    print(f"Is CUDA:       {info.is_cuda}")
    print(f"Is Metal:      {info.is_metal}")
    print(f"Is CPU:        {info.is_cpu}")
    print()
    print("Recommended Configuration:")
    print(f"  MCTS Simulations: {config.mcts_simulations}")
    print(f"  MCTS Max Nodes:   {config.mcts_max_nodes}")
    print(f"  Game Batch Size:  {config.game_batch_size}")
    print(f"  Train Batch Size: {config.train_batch_size}")
    print("=" * 50)

    return info


if __name__ == "__main__":
    # Run device detection when module is executed directly
    print_device_info()
