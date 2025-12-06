"""Shared utilities for distributed training.

This module provides common functionality used across the distributed system.
"""

import signal
import sys
from typing import Callable


def install_shutdown_handler(stop_fn: Callable[[], None]) -> None:
    """Install signal handlers for graceful shutdown.

    Sets up handlers for SIGINT (Ctrl+C) and SIGTERM that call the provided
    stop function and then exit cleanly.

    Args:
        stop_fn: Function to call when shutdown signal is received.
                 Should handle cleanup and stop any running processes.
    """
    def handle_signal(_signum: int, _frame) -> None:
        """Handle shutdown signals."""
        stop_fn()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
