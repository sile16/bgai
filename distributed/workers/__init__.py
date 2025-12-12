"""Distributed worker implementations.

Keep this package import lightweight.

Submodules like `training_worker` import JAX and TurboZero internals and may
pull in symbols that differ across environments. Import worker classes directly
from their modules, e.g.:

    from distributed.workers.game_worker import GameWorker
    from distributed.workers.eval_worker import EvalWorker

No Ray dependency - uses Redis for all coordination.
"""

__all__ = []
