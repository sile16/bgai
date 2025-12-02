"""Training worker for distributed training.

This worker fetches batches of experiences from the Redis replay buffer
and performs gradient updates on the neural network, pushing updated
weights back to the coordinator.

Training is gated on collection milestones:
- Collection runs continuously, filling the buffer
- Training triggers when N new games have been collected
- Training runs until it has processed all available data
- Weights are pushed after each training batch completes
"""

import time
from functools import partial
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import chex
import optax
import ray
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience
from core.training.loss_fns import az_default_loss_fn

from .base_worker import BaseWorker, WorkerStats
from ..serialization import (
    serialize_weights,
    deserialize_weights,
    batch_experiences_to_jax,
)
from ..buffer.redis_buffer import RedisReplayBuffer
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint


@dataclass
class TrainingStats:
    """Statistics for tracking training progress."""
    total_steps: int = 0
    total_batches_trained: int = 0  # Number of training batches (triggered by collection)
    games_at_last_train: int = 0
    experiences_at_last_train: int = 0
    current_batch_steps: int = 0
    current_batch_start_time: float = 0.0
    last_batch_duration: float = 0.0
    last_batch_steps: int = 0
    cumulative_loss: float = 0.0
    loss_count: int = 0

    @property
    def avg_loss(self) -> float:
        return self.cumulative_loss / max(self.loss_count, 1)


@ray.remote
class TrainingWorker(BaseWorker):
    """Ray actor that performs neural network training.

    Samples batches from the Redis replay buffer and performs gradient
    updates, pushing new weights to the coordinator periodically.

    Example:
        >>> coordinator = create_coordinator(config)
        >>> worker = TrainingWorker.remote(coordinator, config={...})
        >>> ray.get(worker.run.remote(num_iterations=10000))
    """

    def __init__(
        self,
        coordinator_handle: ray.actor.ActorHandle,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the training worker.

        Args:
            coordinator_handle: Ray actor handle for the coordinator.
            worker_id: Optional unique worker ID. Auto-generated if not provided.
            config: Configuration dict with keys:
                - train_batch_size: Batch size for training (default: 128)
                - learning_rate: Optimizer learning rate (default: 3e-4)
                - l2_reg_lambda: L2 regularization weight (default: 1e-4)
                - games_per_training_batch: New games to trigger training (default: 10)
                - steps_per_game: Training steps per collected game (default: 10)
                - checkpoint_interval: Steps between checkpoints (default: 1000)
                - min_buffer_size: Minimum buffer size before training (default: 1000)
                - redis_host: Redis server host (default: 'localhost')
                - redis_port: Redis server port (default: 6379)
                - checkpoint_dir: Directory for saving checkpoints (default: '/tmp/distributed_ckpts')
        """
        super().__init__(coordinator_handle, worker_id, config)

        # Training configuration
        self.train_batch_size = self.config.get('train_batch_size', 128)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.l2_reg_lambda = self.config.get('l2_reg_lambda', 1e-4)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        self.min_buffer_size = self.config.get('min_buffer_size', 1000)
        self.checkpoint_dir = self.config.get('checkpoint_dir', '/tmp/distributed_ckpts')

        # Collection-gated training configuration
        # Training triggers after this many new games collected
        self.games_per_training_batch = self.config.get('games_per_training_batch', 10)
        # Number of training steps to run per collected game
        self.steps_per_game = self.config.get('steps_per_game', 10)

        # Surprise-weighted sampling configuration
        # 0 = uniform sampling, 1 = fully surprise-weighted
        self.surprise_weight = self.config.get('surprise_weight', 0.5)

        # Initialize Redis buffer
        redis_host = self.config.get('redis_host', 'localhost')
        redis_port = self.config.get('redis_port', 6379)
        redis_password = self.config.get('redis_password', None)
        self.buffer = RedisReplayBuffer(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            worker_id=self.worker_id,
        )

        # Neural network (lazy initialization)
        self._nn_model = None
        self._train_state = None
        self._env = None

        # Loss function
        self._loss_fn = partial(az_default_loss_fn, l2_reg_lambda=self.l2_reg_lambda)

        # Training state
        self._total_steps = 0
        self._rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

        # Collection-gated training stats
        self._training_stats = TrainingStats()

    @property
    def worker_type(self) -> str:
        return 'training'

    def _setup_environment(self) -> None:
        """Set up the backgammon environment for shape inference."""
        import pgx.backgammon as bg
        self._env = bg.Backgammon(short_game=True)

    def _setup_neural_network(self) -> None:
        """Set up the neural network model."""
        import flax.linen as nn

        class ResidualDenseBlock(nn.Module):
            features: int

            @nn.compact
            def __call__(self, x):
                residual = x
                x = nn.Dense(self.features)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)
                x = nn.Dense(self.features)(x)
                x = nn.LayerNorm()(x)
                return nn.relu(x + residual)

        class ResNetTurboZero(nn.Module):
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6

            @nn.compact
            def __call__(self, x, train: bool = False):
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                value = nn.Dense(1)(x)
                value = jnp.squeeze(value, axis=-1)
                return policy_logits, value

        self._nn_model = ResNetTurboZero(
            self._env.num_actions,
            num_hidden=256,
            num_blocks=6
        )

    def _initialize_train_state(self) -> None:
        """Initialize training state from scratch or from coordinator."""
        # Try to get weights from coordinator
        weights_bytes = self.fetch_model_weights()

        # Create sample input for initialization
        key = jax.random.PRNGKey(42)
        sample_state = self._env.init(key)
        sample_obs = sample_state.observation

        if weights_bytes is not None:
            # Use coordinator weights
            params_dict = deserialize_weights(weights_bytes)
            params = params_dict['params']
            print(f"Worker {self.worker_id}: Loaded model version {self.current_model_version}")
        else:
            # Initialize random weights
            variables = self._nn_model.init(key, sample_obs[None, ...], train=False)
            params = variables['params']
            print(f"Worker {self.worker_id}: Initialized random weights")

        # Create optimizer
        optimizer = optax.adam(self.learning_rate)

        # Create TrainState
        self._train_state = TrainState.create(
            apply_fn=self._nn_model.apply,
            params=params,
            tx=optimizer,
        )

    def _push_weights_to_coordinator(self) -> bool:
        """Push current weights to the coordinator.

        Returns:
            True if weights were pushed successfully.
        """
        try:
            # Serialize weights
            params_dict = {'params': self._train_state.params}
            weights_bytes = serialize_weights(params_dict)

            # Increment version
            new_version = self.current_model_version + 1

            # Push to coordinator
            result = ray.get(self.coordinator.update_model_weights.remote(
                weights_bytes,
                version=new_version,
            ))

            if result.get('status') == 'updated':
                self.current_model_version = new_version
                print(f"Worker {self.worker_id}: Pushed weights version {new_version}")
                return True

            return False

        except Exception as e:
            print(f"Worker {self.worker_id}: Error pushing weights: {e}")
            return False

    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current training state."""
        import os
        import pickle

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'step': self._total_steps,
            'model_version': self.current_model_version,
            'params': jax.device_get(self._train_state.params),
            'opt_state': jax.device_get(self._train_state.opt_state),
        }

        ckpt_path = os.path.join(
            self.checkpoint_dir,
            f"ckpt_{self._total_steps:08d}_v{self.current_model_version}.pkl"
        )

        with open(ckpt_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Worker {self.worker_id}: Saved checkpoint to {ckpt_path}")

        # Clean up old checkpoints (keep last 3)
        ckpt_files = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith('ckpt_') and f.endswith('.pkl')
        ])
        while len(ckpt_files) > 3:
            old_file = os.path.join(self.checkpoint_dir, ckpt_files.pop(0))
            os.remove(old_file)

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(
        self,
        train_state: TrainState,
        batch: BaseExperience,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Perform a single training step.

        Args:
            train_state: Current training state.
            batch: Batch of experiences.

        Returns:
            Tuple of (updated_train_state, metrics_dict).
        """
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (loss, (metrics, updates)), grads = grad_fn(
            train_state.params,
            train_state,
            batch
        )

        # Apply gradients
        new_state = train_state.apply_gradients(grads=grads)

        # Add loss to metrics
        metrics = {**metrics, 'loss': loss}

        return new_state, metrics

    def _sample_and_train(self) -> Optional[Dict[str, float]]:
        """Sample a batch from Redis and perform one training step.

        Returns:
            Metrics dict or None if buffer is not ready.
        """
        # Check buffer size
        buffer_size = self.buffer.get_size()
        if buffer_size < self.min_buffer_size:
            return None

        # Sample batch from Redis with surprise-weighted sampling
        min_version = max(0, self.current_model_version - 10)  # Allow slightly old experiences

        if self.surprise_weight > 0:
            # Use surprise-weighted sampling
            batch_data = self.buffer.sample_batch_surprise_weighted(
                self.train_batch_size,
                surprise_weight=self.surprise_weight,
                min_model_version=min_version,
            )
        else:
            # Uniform sampling
            batch_data = self.buffer.sample_batch(
                self.train_batch_size,
                min_model_version=min_version,
                require_rewards=True,
            )

        if len(batch_data) < self.train_batch_size:
            return None

        # Convert to JAX arrays (returns BaseExperience dataclass)
        try:
            jax_batch = batch_experiences_to_jax(batch_data)
            if jax_batch is None:
                print(f"Worker {self.worker_id}: Empty batch after conversion")
                return None
        except Exception as e:
            print(f"Worker {self.worker_id}: Error converting batch: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Perform training step
        self._train_state, metrics = self._train_step(
            self._train_state,
            jax_batch,
        )

        # Convert metrics to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}

        self._total_steps += 1
        self.stats.training_steps += 1

        return metrics

    def _get_current_games_count(self) -> int:
        """Get current number of completed games in buffer."""
        return self.buffer.redis.llen(self.buffer.EPISODE_LIST)

    def _run_training_batch(self, target_steps: int) -> Dict[str, float]:
        """Run a batch of training steps.

        Args:
            target_steps: Number of training steps to run.

        Returns:
            Dict with averaged metrics from this batch.
        """
        batch_start = time.time()
        batch_metrics = []
        steps_done = 0

        self._training_stats.current_batch_start_time = batch_start
        self._training_stats.current_batch_steps = 0

        while steps_done < target_steps and self.running:
            metrics = self._sample_and_train()

            if metrics is None:
                # Not enough valid experiences, wait briefly
                time.sleep(0.1)
                continue

            batch_metrics.append(metrics)
            steps_done += 1
            self._training_stats.current_batch_steps = steps_done
            self._training_stats.cumulative_loss += metrics.get('loss', 0)
            self._training_stats.loss_count += 1

            # Save checkpoint periodically
            if self._total_steps % self.checkpoint_interval == 0:
                self._save_checkpoint()

        batch_duration = time.time() - batch_start

        # Update stats
        self._training_stats.last_batch_duration = batch_duration
        self._training_stats.last_batch_steps = steps_done
        self._training_stats.total_batches_trained += 1

        # Compute averaged metrics
        if batch_metrics:
            avg_metrics = {
                k: sum(m[k] for m in batch_metrics) / len(batch_metrics)
                for k in batch_metrics[0].keys()
            }
            avg_metrics['batch_duration'] = batch_duration
            avg_metrics['batch_steps'] = steps_done
            avg_metrics['steps_per_sec'] = steps_done / max(batch_duration, 0.001)
            return avg_metrics

        return {'batch_duration': batch_duration, 'batch_steps': 0}

    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main training loop with collection-gated training.

        Training is triggered when enough new games have been collected.
        Collection continues while training runs, so training catches up
        to the collection frontier.

        Args:
            num_iterations: Maximum training steps to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Setup
        print(f"Worker {self.worker_id}: Setting up training...")
        self._setup_environment()
        self._setup_neural_network()
        self._initialize_train_state()

        # Start Prometheus metrics server
        metrics_port = self.config.get('metrics_port', 9200)
        start_metrics_server(metrics_port)
        metrics = get_metrics()

        # Register metrics endpoint for dynamic discovery
        try:
            register_metrics_endpoint(
                self.buffer.redis,
                worker_id=self.worker_id,
                worker_type='training',
                port=metrics_port,
                ttl_seconds=300,
            )
            print(f"Worker {self.worker_id}: Registered metrics endpoint on port {metrics_port}")
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to register metrics endpoint: {e}")

        # Set worker info
        metrics.worker_info.labels(worker_id=self.worker_id).info({
            'type': 'training',
            'batch_size': str(self.train_batch_size),
            'learning_rate': str(self.learning_rate),
        })
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='training'
        ).set(1)

        print(f"Worker {self.worker_id}: Collection-gated training mode")
        print(f"Worker {self.worker_id}: Will train {self.steps_per_game} steps "
              f"for every {self.games_per_training_batch} games collected")

        start_time = time.time()
        last_log_time = start_time

        # Initialize tracking
        self._training_stats.games_at_last_train = self._get_current_games_count()
        self._training_stats.experiences_at_last_train = self.buffer.get_size()

        while self.running:
            if num_iterations >= 0 and self._total_steps >= num_iterations:
                break

            # Check buffer minimum size
            buffer_size = self.buffer.get_size()
            if buffer_size < self.min_buffer_size:
                print(
                    f"Worker {self.worker_id}: Waiting for buffer "
                    f"({buffer_size}/{self.min_buffer_size})..."
                )
                time.sleep(2.0)
                continue

            # Check for new games
            current_games = self._get_current_games_count()
            new_games = current_games - self._training_stats.games_at_last_train

            if new_games < self.games_per_training_batch:
                # Not enough new games, wait
                time.sleep(1.0)

                # Update games since last train metric
                metrics.games_since_last_train.labels(
                    worker_id=self.worker_id
                ).set(new_games)

                # Periodic status log while waiting
                current_time = time.time()
                if current_time - last_log_time >= 30.0:
                    print(
                        f"Worker {self.worker_id}: Waiting for games "
                        f"({new_games}/{self.games_per_training_batch} new games), "
                        f"total_games={current_games}, "
                        f"buffer={buffer_size}, "
                        f"version={self.current_model_version}"
                    )
                    last_log_time = current_time
                continue

            # Calculate how many steps to train
            # Train proportionally to games collected, catching up if behind
            target_steps = new_games * self.steps_per_game

            print(
                f"Worker {self.worker_id}: Training batch triggered! "
                f"{new_games} new games -> {target_steps} training steps"
            )

            # Run training batch
            batch_metrics = self._run_training_batch(target_steps)

            # Push updated weights
            self._push_weights_to_coordinator()

            # Update tracking
            self._training_stats.games_at_last_train = current_games
            self._training_stats.experiences_at_last_train = self.buffer.get_size()

            # Log batch results
            elapsed = time.time() - start_time
            overall_steps_per_sec = self._total_steps / max(elapsed, 0.001)

            # Update Prometheus metrics
            metrics.training_steps_total.labels(
                worker_id=self.worker_id
            ).inc(batch_metrics.get('batch_steps', 0))
            metrics.training_batches_total.labels(
                worker_id=self.worker_id
            ).inc()
            metrics.training_loss.labels(
                worker_id=self.worker_id, loss_type='total'
            ).set(batch_metrics.get('loss', 0))
            metrics.training_steps_per_second.labels(
                worker_id=self.worker_id
            ).set(overall_steps_per_sec)
            metrics.training_batch_duration.labels(
                worker_id=self.worker_id
            ).observe(batch_metrics.get('batch_duration', 0))
            metrics.training_batch_steps.labels(
                worker_id=self.worker_id
            ).observe(batch_metrics.get('batch_steps', 0))
            metrics.buffer_size.set(buffer_size)
            metrics.buffer_games.set(current_games)

            # Record surprise score metrics
            try:
                surprise_stats = self.buffer.get_surprise_stats()
                metrics.episodes_with_surprise.set(surprise_stats['count'])
                metrics.surprise_score_max.set(surprise_stats['max'])
                metrics.surprise_score_mean.set(surprise_stats['mean'])
                metrics.buffer_episodes.set(self.buffer.redis.llen(self.buffer.EPISODE_LIST))
            except Exception:
                pass  # Don't fail training if metrics collection fails

            metrics.model_version.labels(
                worker_id=self.worker_id, worker_type='training'
            ).set(self.current_model_version)
            metrics.weight_updates_total.labels(
                worker_id=self.worker_id
            ).inc()

            # Refresh metrics registration heartbeat
            try:
                register_metrics_endpoint(
                    self.buffer.redis,
                    worker_id=self.worker_id,
                    worker_type='training',
                    port=metrics_port,
                    ttl_seconds=300,
                )
            except Exception as e:
                print(f"Worker {self.worker_id}: Failed to refresh metrics registration: {e}")

            print(
                f"Worker {self.worker_id}: Batch complete! "
                f"step={self._total_steps}, "
                f"loss={batch_metrics.get('loss', 0):.4f}, "
                f"batch_steps={batch_metrics.get('batch_steps', 0)}, "
                f"batch_time={batch_metrics.get('batch_duration', 0):.1f}s, "
                f"batch_steps/s={batch_metrics.get('steps_per_sec', 0):.1f}, "
                f"overall_steps/s={overall_steps_per_sec:.1f}, "
                f"version={self.current_model_version}"
            )

            last_log_time = time.time()

        # Final checkpoint
        if self._total_steps > 0:
            self._push_weights_to_coordinator()
            self._save_checkpoint()

        # Mark worker as stopped
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='training'
        ).set(0)

        # Cleanup
        self.buffer.close()

        return {
            'status': 'completed',
            'total_steps': self._total_steps,
            'total_training_batches': self._training_stats.total_batches_trained,
            'final_model_version': self.current_model_version,
            'duration_seconds': time.time() - start_time,
            'avg_loss': self._training_stats.avg_loss,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics.

        Returns:
            Dict with training stats including collection-gated training info.
        """
        base_stats = self.get_stats()

        # Current buffer state
        buffer_size = self.buffer.get_size() if self.buffer else 0
        current_games = self._get_current_games_count() if self.buffer else 0

        base_stats.update({
            'train_batch_size': self.train_batch_size,
            'learning_rate': self.learning_rate,
            'total_steps': self._total_steps,
            'buffer_size': buffer_size,
            'total_games': current_games,
            # Collection-gated training stats
            'games_per_training_batch': self.games_per_training_batch,
            'steps_per_game': self.steps_per_game,
            'total_training_batches': self._training_stats.total_batches_trained,
            'games_at_last_train': self._training_stats.games_at_last_train,
            'games_since_last_train': current_games - self._training_stats.games_at_last_train,
            'current_batch_steps': self._training_stats.current_batch_steps,
            'last_batch_duration': self._training_stats.last_batch_duration,
            'last_batch_steps': self._training_stats.last_batch_steps,
            'avg_loss': self._training_stats.avg_loss,
        })
        return base_stats

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            True if checkpoint was loaded successfully.
        """
        import pickle

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            # Restore state
            self._total_steps = checkpoint['step']
            self.current_model_version = checkpoint['model_version']

            # Rebuild train state with loaded params
            optimizer = optax.adam(self.learning_rate)
            self._train_state = TrainState.create(
                apply_fn=self._nn_model.apply,
                params=checkpoint['params'],
                tx=optimizer,
            )
            # Restore optimizer state
            self._train_state = self._train_state.replace(
                opt_state=checkpoint['opt_state']
            )

            print(f"Worker {self.worker_id}: Loaded checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Error loading checkpoint: {e}")
            return False
