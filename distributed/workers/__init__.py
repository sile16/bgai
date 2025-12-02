"""Distributed worker implementations.

This module provides standalone worker processes for distributed training:
- GameWorker: Generates self-play games using StochasticMCTS
- TrainingWorker: Performs neural network training on sampled experiences
- EvalWorker: Evaluates models against various baselines
- BaseWorker: Abstract base class for common worker functionality

No Ray dependency - uses Redis for all coordination.
"""

from .base_worker import BaseWorker, WorkerStats
from .game_worker import GameWorker
from .training_worker import TrainingWorker
from .eval_worker import EvalWorker

__all__ = [
    'BaseWorker',
    'WorkerStats',
    'GameWorker',
    'TrainingWorker',
    'EvalWorker',
]
