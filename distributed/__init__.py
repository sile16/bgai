"""Distributed AlphaZero training system using Redis for coordination.

This package provides a distributed training infrastructure that allows
multiple machines to collaborate on training AlphaZero-style agents.

Architecture (Redis-based, no Ray dependency):
- Coordinator: Standalone process managing model versions and cluster state
- GameWorker: Generates self-play games using StochasticMCTS
- TrainingWorker: Performs gradient updates on neural network
- EvalWorker: Evaluates models against baselines
- RedisReplayBuffer: Centralized experience storage

Example usage:
    # Start coordinator on head node
    python -m distributed.cli.main coordinator

    # Start game worker
    python -m distributed.cli.main game-worker

    # Start training worker
    python -m distributed.cli.main training-worker

    # Check status
    python -m distributed.cli.main status
"""

from .device import detect_device, get_device_config, DeviceInfo
from .serialization import (
    serialize_weights,
    deserialize_weights,
    serialize_experience,
    deserialize_experience,
    serialize_warm_tree,
    deserialize_warm_tree,
)
from .coordinator.head_node import (
    Coordinator,
    create_coordinator,
)
from .coordinator.redis_state import (
    RedisStateManager,
    WorkerInfo,
    WorkerStatus,
    RunStatus,
    create_state_manager,
)
from .workers.base_worker import BaseWorker, WorkerStats
from .workers.game_worker import GameWorker
from .workers.training_worker import TrainingWorker
from .workers.eval_worker import EvalWorker
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
    'serialize_warm_tree',
    'deserialize_warm_tree',
    # Coordinator
    'Coordinator',
    'create_coordinator',
    # Redis State
    'RedisStateManager',
    'WorkerInfo',
    'WorkerStatus',
    'RunStatus',
    'create_state_manager',
    # Workers
    'BaseWorker',
    'WorkerStats',
    'GameWorker',
    'TrainingWorker',
    'EvalWorker',
    # Buffer
    'RedisReplayBuffer',
    'BufferStats',
]
