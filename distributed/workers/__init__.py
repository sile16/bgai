"""Distributed worker implementations.

This module provides Ray actors for distributed training:
- GameWorker: Generates self-play games using StochasticMCTS
- TrainingWorker: Performs neural network training on sampled experiences
- BaseWorker: Abstract base class for common worker functionality
"""

from .base_worker import BaseWorker, WorkerStats
from .game_worker import GameWorker
from .training_worker import TrainingWorker

__all__ = [
    'BaseWorker',
    'WorkerStats',
    'GameWorker',
    'TrainingWorker',
]
