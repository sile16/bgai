"""Distributed AlphaZero training system using Ray and Redis.

This package provides a distributed training infrastructure that allows
multiple machines to collaborate on training AlphaZero-style agents.

Architecture:
- Coordinator: Central Ray actor managing workers and model versions
- GameWorker: Generates self-play games using StochasticMCTS
- TrainingWorker: Performs gradient updates on neural network
- RedisReplayBuffer: Centralized experience storage

Example usage:
    # Start coordinator on head node
    python -m distributed.cli.main coordinator --redis-host localhost

    # Start game workers on compute nodes
    python -m distributed.cli.main game-worker --coordinator-address ray://head:10001

    # Start training worker (usually on GPU node)
    python -m distributed.cli.main training-worker --coordinator-address ray://head:10001
"""

from .device import detect_device, get_device_config, DeviceInfo
from .serialization import (
    serialize_weights,
    deserialize_weights,
    serialize_experience,
    deserialize_experience,
)
from .coordinator.head_node import (
    Coordinator,
    create_coordinator,
    get_coordinator,
    get_or_create_coordinator,
)
from .workers.base_worker import BaseWorker, WorkerStats
from .workers.game_worker import GameWorker
from .workers.training_worker import TrainingWorker
from .buffer.redis_buffer import RedisReplayBuffer, BufferStats

__all__ = [
    # Device detection
    'detect_device',
    'get_device_config',
    'DeviceInfo',
    # Serialization
    'serialize_weights',
    'deserialize_weights',
    'serialize_experience',
    'deserialize_experience',
    # Coordinator
    'Coordinator',
    'create_coordinator',
    'get_coordinator',
    'get_or_create_coordinator',
    # Workers
    'BaseWorker',
    'WorkerStats',
    'GameWorker',
    'TrainingWorker',
    # Buffer
    'RedisReplayBuffer',
    'BufferStats',
]
