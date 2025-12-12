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

# Keep package import lightweight.
#
# Many submodules (device detection, workers, evaluators) import JAX and may
# initialize CUDA. CLI commands like `status` should be able to run even when
# the GPU is fully occupied, so avoid importing heavy modules at import time.
#
# Import needed symbols from their specific modules instead of from
# `distributed` directly.

__all__ = []
