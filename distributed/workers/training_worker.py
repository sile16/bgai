"""Training worker for distributed training.

This worker fetches batches of experiences from the Redis replay buffer
and performs gradient updates on the neural network, pushing updated
weights back to the coordinator.
"""

import time
from functools import partial
from typing import Dict, Any, Optional, Tuple

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
                - weight_push_interval: Steps between weight pushes (default: 10)
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
        self.weight_push_interval = self.config.get('weight_push_interval', 10)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        self.min_buffer_size = self.config.get('min_buffer_size', 1000)
        self.checkpoint_dir = self.config.get('checkpoint_dir', '/tmp/distributed_ckpts')

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

        # Sample batch from Redis
        batch_data = self.buffer.sample_batch(
            self.train_batch_size,
            min_model_version=max(0, self.current_model_version - 10),  # Allow slightly old experiences
            require_rewards=True,
        )

        if len(batch_data) < self.train_batch_size:
            return None

        # Convert to JAX arrays
        try:
            jax_batch = batch_experiences_to_jax(batch_data)
        except Exception as e:
            print(f"Worker {self.worker_id}: Error converting batch: {e}")
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

    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main training loop.

        Args:
            num_iterations: Number of training steps to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Setup
        print(f"Worker {self.worker_id}: Setting up training...")
        self._setup_environment()
        self._setup_neural_network()
        self._initialize_train_state()

        print(f"Worker {self.worker_id}: Starting training loop...")

        start_time = time.time()
        last_log_time = start_time
        accumulated_metrics = []
        wait_logged = False

        while self.running:
            if num_iterations >= 0 and self._total_steps >= num_iterations:
                break

            # Train step
            metrics = self._sample_and_train()

            if metrics is None:
                # Buffer not ready, wait
                if not wait_logged:
                    buffer_size = self.buffer.get_size()
                    print(
                        f"Worker {self.worker_id}: Waiting for buffer "
                        f"({buffer_size}/{self.min_buffer_size})..."
                    )
                    wait_logged = True
                time.sleep(1.0)
                continue

            wait_logged = False
            accumulated_metrics.append(metrics)

            # Push weights periodically
            if self._total_steps % self.weight_push_interval == 0:
                self._push_weights_to_coordinator()

            # Save checkpoint periodically
            if self._total_steps % self.checkpoint_interval == 0:
                self._save_checkpoint()

            # Log metrics periodically
            current_time = time.time()
            if current_time - last_log_time >= 10.0:  # Log every 10 seconds
                if accumulated_metrics:
                    avg_metrics = {
                        k: sum(m[k] for m in accumulated_metrics) / len(accumulated_metrics)
                        for k in accumulated_metrics[0].keys()
                    }
                    elapsed = current_time - start_time
                    steps_per_sec = self._total_steps / max(elapsed, 0.001)

                    print(
                        f"Worker {self.worker_id}: "
                        f"step={self._total_steps}, "
                        f"loss={avg_metrics.get('loss', 0):.4f}, "
                        f"steps/s={steps_per_sec:.1f}, "
                        f"version={self.current_model_version}"
                    )
                    accumulated_metrics = []
                last_log_time = current_time

        # Final checkpoint
        if self._total_steps > 0:
            self._push_weights_to_coordinator()
            self._save_checkpoint()

        # Cleanup
        self.buffer.close()

        return {
            'status': 'completed',
            'total_steps': self._total_steps,
            'final_model_version': self.current_model_version,
            'duration_seconds': time.time() - start_time,
        }

    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics.

        Returns:
            Dict with training stats.
        """
        base_stats = self.get_stats()
        base_stats.update({
            'train_batch_size': self.train_batch_size,
            'learning_rate': self.learning_rate,
            'total_steps': self._total_steps,
            'buffer_size': self.buffer.get_size() if self.buffer else 0,
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
