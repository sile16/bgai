#!/usr/bin/env python
"""CLI for distributed backgammon AI training.

No Ray dependency - workers run as standalone Python processes
coordinated via Redis.

Usage:
    # Start coordinator (head node)
    python -m distributed.cli.main coordinator

    # Start game worker
    python -m distributed.cli.main game-worker

    # Start training worker
    python -m distributed.cli.main training-worker

    # Start evaluation worker
    python -m distributed.cli.main eval-worker

    # Check cluster status
    python -m distributed.cli.main status

    # Training run management
    python -m distributed.cli.main runs list
    python -m distributed.cli.main runs start
    python -m distributed.cli.main runs pause
    python -m distributed.cli.main runs resume
    python -m distributed.cli.main runs stop
"""

import argparse
import os
import sys
import time
from typing import Optional

from .config_loader import (
    load_yaml_config,
    detect_device_type,
    get_coordinator_config,
    get_game_worker_config,
    get_training_worker_config,
    get_eval_worker_config,
    print_config_summary,
)


def configure_gpu_fraction(num_gpus: float) -> None:
    """Configure JAX to use a fraction of GPU memory.

    IMPORTANT: Must be called BEFORE any JAX imports.

    Args:
        num_gpus: Fraction of GPU to use (0.0-1.0).
                  Use 0.5 to share GPU between two processes.
    """
    if num_gpus < 1.0:
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(num_gpus)
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print(f"GPU memory fraction set to {num_gpus}")


def start_coordinator(args):
    """Start the coordinator (head node)."""
    from ..coordinator.head_node import Coordinator

    # Load configuration from file
    yaml_config = load_yaml_config(args.config_file)
    file_config = get_coordinator_config(yaml_config)

    # Merge: file config <- CLI overrides (CLI takes precedence if specified)
    config = file_config.copy()
    if args.redis_host != 'localhost':  # Only override if explicitly set
        config['redis_host'] = args.redis_host
    if args.redis_port != 6379:
        config['redis_port'] = args.redis_port
    if args.redis_password is not None:
        config['redis_password'] = args.redis_password
    if args.mcts_simulations != 100:
        config['mcts_simulations'] = args.mcts_simulations
    if args.mcts_max_nodes != 400:
        config['mcts_max_nodes'] = args.mcts_max_nodes
    if args.status_interval != 10:
        config['status_interval'] = args.status_interval

    print(f"Creating coordinator with config:")
    for k, v in config.items():
        if k != 'device_configs':  # Don't print the large device_configs dict
            print(f"  {k}: {v}")

    # Create and start coordinator
    coordinator = Coordinator(config)

    print("\nCoordinator running. Press Ctrl+C to stop.")
    try:
        # This blocks and runs the status loop
        coordinator.start(blocking=True)
    except KeyboardInterrupt:
        print("\nShutting down coordinator...")
        coordinator.stop()


def start_game_worker(args):
    """Start a game generation worker."""
    # Configure GPU fraction BEFORE importing JAX-dependent modules
    configure_gpu_fraction(args.num_gpus)

    from ..workers.game_worker import GameWorker

    # Load configuration from file
    yaml_config = load_yaml_config(args.config_file)
    device_type = detect_device_type()

    # Get device-specific config with optional batch size override
    batch_override = args.batch_size if args.batch_size != 16 else None
    config = get_game_worker_config(yaml_config, device_type, batch_override)

    # CLI overrides
    if args.redis_host != 'localhost':
        config['redis_host'] = args.redis_host
    if args.redis_port != 6379:
        config['redis_port'] = args.redis_port
    if args.redis_password is not None:
        config['redis_password'] = args.redis_password
    if args.mcts_simulations != 100:
        config['num_simulations'] = args.mcts_simulations
    if args.mcts_max_nodes != 400:
        config['max_nodes'] = args.mcts_max_nodes
    if args.temperature != 1.0:
        config['temperature'] = args.temperature
    if args.max_episode_steps != 500:
        config['max_episode_steps'] = args.max_episode_steps
    if args.metrics_port != 9100:
        config['metrics_port'] = args.metrics_port

    # Print config summary
    print_config_summary(yaml_config, device_type)

    print(f"Starting game worker with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create and run worker
    worker = GameWorker(
        config=config,
        worker_id=args.worker_id,
    )

    print("\nGame worker running. Press Ctrl+C to stop.")
    try:
        result = worker.run(num_iterations=args.num_iterations)
        print(f"Worker finished: {result}")
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()


def start_training_worker(args):
    """Start a training worker."""
    # Configure GPU fraction BEFORE importing JAX-dependent modules
    configure_gpu_fraction(args.num_gpus)

    from ..workers.training_worker import TrainingWorker

    # Load configuration from file
    yaml_config = load_yaml_config(args.config_file)
    device_type = detect_device_type()

    # Get device-specific config with optional batch size override
    batch_override = args.batch_size if args.batch_size != 128 else None
    head_ip = getattr(args, 'head_ip', None)
    config = get_training_worker_config(yaml_config, device_type, batch_override, head_ip=head_ip)

    # CLI overrides
    if args.redis_host != 'localhost':
        config['redis_host'] = args.redis_host
    if args.redis_port != 6379:
        config['redis_port'] = args.redis_port
    if args.redis_password is not None:
        config['redis_password'] = args.redis_password
    if args.learning_rate != 3e-4:
        config['learning_rate'] = args.learning_rate
    if args.l2_reg != 1e-4:
        config['l2_reg_lambda'] = args.l2_reg
    if args.games_per_batch != 10:
        config['games_per_training_batch'] = args.games_per_batch
    if args.steps_per_game != 10:
        config['steps_per_game'] = args.steps_per_game
    if args.surprise_weight != 0.5:
        config['surprise_weight'] = args.surprise_weight
    if args.checkpoint_interval != 1000:
        config['checkpoint_interval'] = args.checkpoint_interval
    if args.min_buffer_size != 1000:
        config['min_buffer_size'] = args.min_buffer_size
    if args.checkpoint_dir:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.metrics_port != 9200:
        config['metrics_port'] = args.metrics_port
    if args.mlflow_uri:
        config['mlflow_tracking_uri'] = args.mlflow_uri
    if args.mlflow_experiment:
        config['mlflow_experiment_name'] = args.mlflow_experiment

    # Print config summary
    print_config_summary(yaml_config, device_type)

    print(f"Starting training worker with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create and run worker
    worker = TrainingWorker(
        config=config,
        worker_id=args.worker_id,
    )

    print("\nTraining worker running. Press Ctrl+C to stop.")
    try:
        result = worker.run(num_iterations=args.num_iterations)
        print(f"Worker finished: {result}")
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()


def start_eval_worker(args):
    """Start an evaluation worker."""
    # Configure GPU fraction BEFORE importing JAX-dependent modules
    configure_gpu_fraction(args.num_gpus)

    from ..workers.eval_worker import EvalWorker

    # Load configuration from file
    yaml_config = load_yaml_config(args.config_file)
    device_type = detect_device_type()

    # Get device-specific config with optional batch size override
    batch_override = args.batch_size if args.batch_size != 16 else None
    head_ip = getattr(args, 'head_ip', None)
    config = get_eval_worker_config(yaml_config, device_type, batch_override, head_ip=head_ip)

    # CLI overrides
    if args.redis_host != 'localhost':
        config['redis_host'] = args.redis_host
    if args.redis_port != 6379:
        config['redis_port'] = args.redis_port
    if args.redis_password is not None:
        config['redis_password'] = args.redis_password
    if args.mcts_simulations != 200:
        config['num_simulations'] = args.mcts_simulations
    if args.mcts_max_nodes != 800:
        config['max_nodes'] = args.mcts_max_nodes
    if args.metrics_port != 9300:
        config['metrics_port'] = args.metrics_port

    # Eval-specific settings
    config['eval_games'] = args.eval_games
    config['eval_interval'] = args.eval_interval

    if args.eval_types:
        config['eval_types'] = [t.strip() for t in args.eval_types.split(',')]

    # Print config summary
    print_config_summary(yaml_config, device_type)

    print(f"Starting evaluation worker with config:")
    for k, v in config.items():
        if v is not None:
            print(f"  {k}: {v}")

    # Create and run worker
    worker = EvalWorker(
        config=config,
        worker_id=args.worker_id,
    )

    print("\nEvaluation worker running. Press Ctrl+C to stop.")
    try:
        result = worker.run(num_iterations=args.num_iterations)
        print(f"Worker finished: {result}")
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()


def show_status(args):
    """Show cluster status from Redis."""
    from ..coordinator.redis_state import create_state_manager

    # Load configuration from file
    yaml_config = load_yaml_config(args.config_file)

    config = {
        'redis_host': args.redis_host if args.redis_host != 'localhost' else yaml_config.get('redis', {}).get('host', 'localhost'),
        'redis_port': args.redis_port if args.redis_port != 6379 else yaml_config.get('redis', {}).get('port', 6379),
        'redis_password': args.redis_password or yaml_config.get('redis', {}).get('password'),
    }

    try:
        state = create_state_manager(config)
        if not state.ping():
            print(f"ERROR: Cannot connect to Redis at {config['redis_host']}:{config['redis_port']}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        sys.exit(1)

    status = state.get_cluster_status()
    workers = status['workers']

    print("\n=== Cluster Status ===")
    print(f"Model version: {status['model_version']}")
    print(f"Run ID: {status['run_id'] or 'None'}")
    print(f"Run status: {status['run_status']}")
    print(f"Total games generated: {status['total_games_generated']}")
    print(f"Total training steps: {status['total_training_steps']}")

    print(f"\n=== Workers ===")
    print(f"Game workers: {workers['game']}")
    print(f"Training workers: {workers['training']}")
    print(f"Evaluation workers: {workers['eval']}")
    print(f"Total: {workers['total']}")

    if status['active_workers']:
        print(f"\n=== Active Workers ===")
        for w in status['active_workers']:
            print(
                f"  {w['worker_id']}: type={w['worker_type']}, "
                f"device={w['device_type']}, status={w['status']}, "
                f"games={w['games_generated']}, model_v{w['model_version']}"
            )
    else:
        print(f"\n(No active workers)")


def manage_runs(args):
    """Manage training runs."""
    from ..coordinator.redis_state import create_state_manager, RunStatus

    # Load configuration
    yaml_config = load_yaml_config(args.config_file)

    config = {
        'redis_host': args.redis_host if args.redis_host != 'localhost' else yaml_config.get('redis', {}).get('host', 'localhost'),
        'redis_port': args.redis_port if args.redis_port != 6379 else yaml_config.get('redis', {}).get('port', 6379),
        'redis_password': args.redis_password or yaml_config.get('redis', {}).get('password'),
    }

    try:
        state = create_state_manager(config)
        if not state.ping():
            print(f"ERROR: Cannot connect to Redis")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        sys.exit(1)

    if args.run_command == 'list':
        run_id = state.get_run_id()
        run_status = state.get_run_status()
        print(f"Current run ID: {run_id or 'None'}")
        print(f"Status: {run_status}")

    elif args.run_command == 'start':
        import uuid
        run_id = args.run_id or f"run_{uuid.uuid4().hex[:8]}"
        current_status = state.get_run_status()
        if current_status == RunStatus.RUNNING.value:
            print(f"ERROR: Training already running (run_id: {state.get_run_id()})")
            sys.exit(1)
        state.set_run_id(run_id)
        state.set_run_status(RunStatus.RUNNING)
        print(f"Started training run: {run_id}")

    elif args.run_command == 'pause':
        if state.get_run_status() != RunStatus.RUNNING.value:
            print("ERROR: No active training run to pause")
            sys.exit(1)
        state.set_run_status(RunStatus.PAUSED)
        print(f"Paused training run: {state.get_run_id()}")

    elif args.run_command == 'resume':
        if state.get_run_status() != RunStatus.PAUSED.value:
            print("ERROR: No paused training run to resume")
            sys.exit(1)
        state.set_run_status(RunStatus.RUNNING)
        print(f"Resumed training run: {state.get_run_id()}")

    elif args.run_command == 'stop':
        run_id = state.get_run_id()
        state.set_run_status(RunStatus.STOPPED)
        print(f"Stopped training run: {run_id}")

    elif args.run_command == 'reset':
        if args.confirm:
            state.reset_cluster_state()
            print("Cluster state reset (all workers deregistered, model version reset)")
        else:
            print("WARNING: This will reset all cluster state!")
            print("Use --confirm to proceed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Distributed backgammon AI training (Redis-based, no Ray)',
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
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
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
        '--status-interval',
        type=float,
        default=10.0,
        help='Status print interval in seconds (default: 10)'
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
        '--dashboard',
        action='store_true',
        help='Enable dashboard mode (reserved for future use)'
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
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
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
        '--metrics-port',
        type=int,
        default=9100,
        help='Prometheus metrics port (default: 9100)'
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
        default=1.0,
        help='GPU fraction to use (default: 1.0, use 0.5 to share GPU)'
    )
    game_parser.add_argument(
        '--head-ip',
        type=str,
        default=None,
        help='Head node IP for MLflow/services (auto-detected if not provided)'
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
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
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
        '--games-per-batch',
        type=int,
        default=10,
        help='New games required to trigger a training batch (default: 10)'
    )
    train_parser.add_argument(
        '--steps-per-game',
        type=int,
        default=10,
        help='Training steps to run per collected game (default: 10)'
    )
    train_parser.add_argument(
        '--surprise-weight',
        type=float,
        default=0.5,
        help='Weight for surprise-based sampling (0=uniform, 1=fully weighted, default: 0.5)'
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
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Checkpoint directory (default: checkpoints)'
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
        '--metrics-port',
        type=int,
        default=9200,
        help='Prometheus metrics port (default: 9200)'
    )
    train_parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: None, MLflow disabled)'
    )
    train_parser.add_argument(
        '--mlflow-experiment',
        type=str,
        default='bgai-training',
        help='MLflow experiment name (default: bgai-training)'
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
        default=1.0,
        help='GPU fraction to use (default: 1.0, use 0.5 to share GPU)'
    )
    train_parser.add_argument(
        '--head-ip',
        type=str,
        default=None,
        help='Head node IP for MLflow/services (auto-detected if not provided)'
    )
    train_parser.set_defaults(func=start_training_worker)

    # =========================================================================
    # Evaluation worker command
    # =========================================================================
    eval_parser = subparsers.add_parser(
        'eval-worker',
        help='Start an evaluation worker'
    )
    eval_parser.add_argument(
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
    )
    eval_parser.add_argument(
        '--worker-id',
        type=str,
        default=None,
        help='Worker ID (auto-generated if not provided)'
    )
    eval_parser.add_argument(
        '--eval-games',
        type=int,
        default=100,
        help='Number of games per evaluation (default: 100)'
    )
    eval_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of parallel games (default: 16)'
    )
    eval_parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=200,
        help='MCTS simulations per move for evaluation (default: 200)'
    )
    eval_parser.add_argument(
        '--mcts-max-nodes',
        type=int,
        default=800,
        help='Maximum MCTS tree nodes (default: 800)'
    )
    eval_parser.add_argument(
        '--eval-interval',
        type=int,
        default=300,
        help='Seconds between evaluation checks (default: 300)'
    )
    eval_parser.add_argument(
        '--eval-types',
        type=str,
        default=None,
        help='Comma-separated list of eval types: gnubg,random,self_play (default: all)'
    )
    eval_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    eval_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    eval_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    eval_parser.add_argument(
        '--metrics-port',
        type=int,
        default=9300,
        help='Prometheus metrics port (default: 9300)'
    )
    eval_parser.add_argument(
        '--num-iterations',
        type=int,
        default=-1,
        help='Number of evaluation runs (-1 for infinite, default: -1)'
    )
    eval_parser.add_argument(
        '--num-gpus',
        type=float,
        default=1.0,
        help='GPU fraction to use (default: 1.0, use 0.5 to share GPU)'
    )
    eval_parser.add_argument(
        '--head-ip',
        type=str,
        default=None,
        help='Head node IP for MLflow/services (auto-detected if not provided)'
    )
    eval_parser.set_defaults(func=start_eval_worker)

    # =========================================================================
    # Status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        'status',
        help='Show cluster status'
    )
    status_parser.add_argument(
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
    )
    status_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    status_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    status_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    status_parser.set_defaults(func=show_status)

    # =========================================================================
    # Runs management command
    # =========================================================================
    runs_parser = subparsers.add_parser(
        'runs',
        help='Manage training runs'
    )
    runs_parser.add_argument(
        'run_command',
        choices=['list', 'start', 'pause', 'resume', 'stop', 'reset'],
        help='Run management command'
    )
    runs_parser.add_argument(
        '--config-file', '-c',
        type=str,
        default=None,
        help='Path to config file (default: configs/distributed.yaml)'
    )
    runs_parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for start command (auto-generated if not provided)'
    )
    runs_parser.add_argument(
        '--redis-host',
        type=str,
        default='localhost',
        help='Redis host (default: localhost)'
    )
    runs_parser.add_argument(
        '--redis-port',
        type=int,
        default=6379,
        help='Redis port (default: 6379)'
    )
    runs_parser.add_argument(
        '--redis-password',
        type=str,
        default=None,
        help='Redis password (default: None)'
    )
    runs_parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm destructive operations like reset'
    )
    runs_parser.set_defaults(func=manage_runs)

    # Parse and execute
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
