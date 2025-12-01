#!/usr/bin/env python
"""CLI for distributed backgammon AI training.

Usage:
    # Start coordinator (head node)
    python -m distributed.cli.main coordinator --redis-host localhost

    # Start game worker
    python -m distributed.cli.main game-worker --coordinator-address ray://head:10001

    # Start training worker
    python -m distributed.cli.main training-worker --coordinator-address ray://head:10001

    # Check cluster status
    python -m distributed.cli.main status --coordinator-address ray://head:10001
"""

import argparse
import sys
import time
from typing import Optional

import ray


def start_coordinator(args):
    """Start the coordinator (head node)."""
    import os
    from ..coordinator.head_node import create_coordinator

    # Get project directory for runtime environment
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Initialize Ray (head node)
    print("Initializing Ray head node...")
    print(f"Project directory: {project_dir}")
    ray.init(
        address=args.ray_address if args.ray_address else None,
        dashboard_host='0.0.0.0' if args.dashboard else None,
        include_dashboard=args.dashboard,
        namespace="bgai",  # Use consistent namespace for all workers
        runtime_env={"env_vars": {"PYTHONPATH": project_dir}},
    )

    config = {
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'redis_password': args.redis_password,
        'heartbeat_timeout': args.heartbeat_timeout,
        'heartbeat_interval': args.heartbeat_interval,
        'mcts_simulations': args.mcts_simulations,
        'mcts_max_nodes': args.mcts_max_nodes,
        'train_batch_size': args.train_batch_size,
        'game_batch_size': args.game_batch_size,
        'learning_rate': args.learning_rate,
    }

    print(f"Creating coordinator with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    coordinator = create_coordinator(config, name='coordinator')
    print(f"Coordinator started. Dashboard available at http://127.0.0.1:8265")

    # Keep running
    print("\nCoordinator running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            # Print status periodically
            status = ray.get(coordinator.get_cluster_status.remote())
            counts = ray.get(coordinator.get_worker_count.remote())
            print(
                f"Status: model_v{status['model_version']}, "
                f"games: {counts['game']}, training: {counts['training']}, "
                f"total_games: {status['total_games_generated']}, "
                f"train_steps: {status['total_training_steps']}"
            )
    except KeyboardInterrupt:
        print("\nShutting down coordinator...")
        ray.get(coordinator.shutdown.remote())
        ray.shutdown()


def start_game_worker(args):
    """Start a game generation worker."""
    import os
    from ..workers.game_worker import GameWorker
    from ..coordinator.head_node import get_coordinator

    # Get project directory for runtime environment
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Connect to Ray cluster
    print(f"Connecting to Ray cluster at {args.coordinator_address}...")
    ray.init(
        address=args.coordinator_address,
        namespace="bgai",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": project_dir,
                # Limit JAX memory to allow multiple workers on same GPU
                # Based on benchmarks: ~2GB per worker, 25% = 6GB headroom on 24GB GPU
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
            }
        },
    )

    # Get coordinator handle
    coordinator = get_coordinator('coordinator')
    if coordinator is None:
        print("ERROR: Could not find coordinator. Make sure it's running.")
        sys.exit(1)

    config = {
        'batch_size': args.batch_size,
        'num_simulations': args.mcts_simulations,
        'max_nodes': args.mcts_max_nodes,
        'temperature': args.temperature,
        'max_episode_steps': args.max_episode_steps,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'redis_password': args.redis_password,
    }

    print(f"Starting game worker with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create worker with optional GPU resources
    num_gpus = args.num_gpus if hasattr(args, 'num_gpus') else 0
    if num_gpus > 0:
        print(f"Requesting {num_gpus} GPU(s)")
        worker = GameWorker.options(num_gpus=num_gpus).remote(
            coordinator_handle=coordinator,
            worker_id=args.worker_id,
            config=config,
        )
    else:
        worker = GameWorker.remote(
            coordinator_handle=coordinator,
            worker_id=args.worker_id,
            config=config,
        )

    # Run worker
    print("\nGame worker running. Press Ctrl+C to stop.")
    try:
        result = ray.get(worker.run.remote(num_iterations=args.num_iterations))
        print(f"Worker finished: {result}")
    except KeyboardInterrupt:
        print("\nStopping worker...")
        ray.get(worker.stop.remote())

    ray.shutdown()


def start_training_worker(args):
    """Start a training worker."""
    import os
    from ..workers.training_worker import TrainingWorker
    from ..coordinator.head_node import get_coordinator

    # Get project directory for runtime environment
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Connect to Ray cluster
    print(f"Connecting to Ray cluster at {args.coordinator_address}...")
    ray.init(
        address=args.coordinator_address,
        namespace="bgai",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": project_dir,
                # Limit JAX memory to allow multiple workers on same GPU
                # Based on benchmarks: ~2GB per worker, 25% = 6GB headroom on 24GB GPU
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
            }
        },
    )

    # Get coordinator handle
    coordinator = get_coordinator('coordinator')
    if coordinator is None:
        print("ERROR: Could not find coordinator. Make sure it's running.")
        sys.exit(1)

    config = {
        'train_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'l2_reg_lambda': args.l2_reg,
        'weight_push_interval': args.weight_push_interval,
        'checkpoint_interval': args.checkpoint_interval,
        'min_buffer_size': args.min_buffer_size,
        'min_new_experiences': args.min_new_experiences,
        'checkpoint_dir': args.checkpoint_dir,
        'redis_host': args.redis_host,
        'redis_port': args.redis_port,
        'redis_password': args.redis_password,
    }

    print(f"Starting training worker with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create worker with optional GPU resources
    num_gpus = args.num_gpus if hasattr(args, 'num_gpus') else 0
    if num_gpus > 0:
        print(f"Requesting {num_gpus} GPU(s)")
        worker = TrainingWorker.options(num_gpus=num_gpus).remote(
            coordinator_handle=coordinator,
            worker_id=args.worker_id,
            config=config,
        )
    else:
        worker = TrainingWorker.remote(
            coordinator_handle=coordinator,
            worker_id=args.worker_id,
            config=config,
        )

    # Run worker
    print("\nTraining worker running. Press Ctrl+C to stop.")
    try:
        result = ray.get(worker.run.remote(num_iterations=args.num_iterations))
        print(f"Worker finished: {result}")
    except KeyboardInterrupt:
        print("\nStopping worker...")
        ray.get(worker.stop.remote())

    ray.shutdown()


def show_status(args):
    """Show cluster status."""
    import os
    from ..coordinator.head_node import get_coordinator

    # Get project directory for runtime environment
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Connect to Ray cluster
    ray.init(
        address=args.coordinator_address,
        namespace="bgai",
        runtime_env={"env_vars": {"PYTHONPATH": project_dir}},
    )

    coordinator = get_coordinator('coordinator')
    if coordinator is None:
        print("ERROR: Could not find coordinator. Make sure it's running.")
        sys.exit(1)

    status = ray.get(coordinator.get_cluster_status.remote())
    counts = ray.get(coordinator.get_worker_count.remote())

    print("\n=== Cluster Status ===")
    print(f"Model version: {status['model_version']}")
    print(f"Uptime: {status['uptime_seconds']:.1f}s")
    print(f"Total games: {status['total_games_generated']}")
    print(f"Total training steps: {status['total_training_steps']}")

    print(f"\n=== Workers ===")
    print(f"Game workers: {counts['game']}")
    print(f"Training workers: {counts['training']}")
    print(f"Evaluation workers: {counts['evaluation']}")
    print(f"Disconnected: {counts['disconnected']}")

    print(f"\n=== Active Workers ===")
    for w in status['active_workers']:
        print(
            f"  {w['id']}: type={w['type']}, device={w['device']}, "
            f"status={w['status']}, games={w['games']}, "
            f"model_v{w['model_version']}"
        )

    ray.shutdown()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Distributed backgammon AI training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # =========================================================================
    # Coordinator command
    # =========================================================================
    coord_parser = subparsers.add_parser(
        'coordinator',
        help='Start the coordinator (head node)'
    )
    coord_parser.add_argument(
        '--ray-address',
        type=str,
        default=None,
        help='Ray cluster address (default: start new cluster)'
    )
    coord_parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Enable Ray dashboard'
    )
    coord_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    coord_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    coord_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    coord_parser.add_argument(
        '--heartbeat-timeout',
        type=float,
        default=30.0,
        help='Worker heartbeat timeout in seconds (default: 30)'
    )
    coord_parser.add_argument(
        '--heartbeat-interval',
        type=float,
        default=10.0,
        help='Worker heartbeat interval in seconds (default: 10)'
    )
    coord_parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=100,
        help='MCTS simulations per move (default: 100)'
    )
    coord_parser.add_argument(
        '--mcts-max-nodes',
        type=int,
        default=400,
        help='Maximum MCTS tree nodes (default: 400)'
    )
    coord_parser.add_argument(
        '--train-batch-size',
        type=int,
        default=128,
        help='Training batch size (default: 128)'
    )
    coord_parser.add_argument(
        '--game-batch-size',
        type=int,
        default=16,
        help='Game worker batch size (default: 16)'
    )
    coord_parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    coord_parser.set_defaults(func=start_coordinator)

    # =========================================================================
    # Game worker command
    # =========================================================================
    game_parser = subparsers.add_parser(
        'game-worker',
        help='Start a game generation worker'
    )
    game_parser.add_argument(
        '--coordinator-address',
        type=str,
        required=True,
        help='Ray cluster address (e.g., ray://host:10001)'
    )
    game_parser.add_argument(
        '--worker-id',
        type=str,
        default=None,
        help='Worker ID (auto-generated if not provided)'
    )
    game_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of parallel environments (default: 16)'
    )
    game_parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=100,
        help='MCTS simulations per move (default: 100)'
    )
    game_parser.add_argument(
        '--mcts-max-nodes',
        type=int,
        default=400,
        help='Maximum MCTS tree nodes (default: 400)'
    )
    game_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='MCTS temperature (default: 1.0)'
    )
    game_parser.add_argument(
        '--max-episode-steps',
        type=int,
        default=500,
        help='Maximum steps per episode (default: 500)'
    )
    game_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    game_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    game_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    game_parser.add_argument(
        '--num-iterations',
        type=int,
        default=-1,
        help='Number of iterations (-1 for infinite, default: -1)'
    )
    game_parser.add_argument(
        '--num-gpus',
        type=float,
        default=0,
        help='Number of GPUs to request (0 for CPU only, default: 0)'
    )
    game_parser.set_defaults(func=start_game_worker)

    # =========================================================================
    # Training worker command
    # =========================================================================
    train_parser = subparsers.add_parser(
        'training-worker',
        help='Start a training worker'
    )
    train_parser.add_argument(
        '--coordinator-address',
        type=str,
        required=True,
        help='Ray cluster address (e.g., ray://host:10001)'
    )
    train_parser.add_argument(
        '--worker-id',
        type=str,
        default=None,
        help='Worker ID (auto-generated if not provided)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Training batch size (default: 128)'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    train_parser.add_argument(
        '--l2-reg',
        type=float,
        default=1e-4,
        help='L2 regularization weight (default: 1e-4)'
    )
    train_parser.add_argument(
        '--weight-push-interval',
        type=int,
        default=10,
        help='Steps between weight pushes (default: 10)'
    )
    train_parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=1000,
        help='Steps between checkpoints (default: 1000)'
    )
    train_parser.add_argument(
        '--min-buffer-size',
        type=int,
        default=1000,
        help='Minimum buffer size before training (default: 1000)'
    )
    train_parser.add_argument(
        '--min-new-experiences',
        type=int,
        default=256,
        help='Minimum new experiences before pushing weight updates (default: 256)'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/tmp/distributed_ckpts',
        help='Checkpoint directory (default: /tmp/distributed_ckpts)'
    )
    train_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    train_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    train_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    train_parser.add_argument(
        '--num-iterations',
        type=int,
        default=-1,
        help='Number of training steps (-1 for infinite, default: -1)'
    )
    train_parser.add_argument(
        '--num-gpus',
        type=float,
        default=0,
        help='Number of GPUs to request (0 for CPU only, default: 0)'
    )
    train_parser.set_defaults(func=start_training_worker)

    # =========================================================================
    # Status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        'status',
        help='Show cluster status'
    )
    status_parser.add_argument(
        '--coordinator-address',
        type=str,
        required=True,
        help='Ray cluster address (e.g., ray://host:10001)'
    )
    status_parser.set_defaults(func=show_status)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
