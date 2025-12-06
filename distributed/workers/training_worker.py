"""Training worker for distributed training.

This worker fetches batches of experiences from the Redis replay buffer
and performs gradient updates on the neural network, pushing updated
weights back to Redis for other workers to fetch.

Training is gated on collection milestones:
- Collection runs continuously, filling the buffer
- Training triggers when N new games have been collected
- Training runs until it has processed all available data
- Weights are pushed after each training batch completes

No Ray dependency - uses Redis for all coordination.
"""

import time
import os
from functools import partial
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import math
import jax
import jax.numpy as jnp
import chex
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience
from bgai.endgame.packed_table import BearoffLookup, pack_upper, solve_size_to_n
from core.training.loss_fns import az_default_loss_fn

# Bearoff table for perfect endgame values (imports lazy-loaded in methods)

from .base_worker import BaseWorker, WorkerStats
from ..serialization import (
    serialize_weights,
    deserialize_weights,
    batch_experiences_to_jax,
    serialize_warm_tree,
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
    # Bearoff/endgame statistics
    bearoff_experiences: int = 0
    non_bearoff_experiences: int = 0
    cumulative_bearoff_value_loss: float = 0.0
    cumulative_non_bearoff_value_loss: float = 0.0
    last_batch_duration: float = 0.0
    last_batch_steps: int = 0
    cumulative_loss: float = 0.0
    loss_count: int = 0

    @property
    def avg_loss(self) -> float:
        return self.cumulative_loss / max(self.loss_count, 1)


class TrainingWorker(BaseWorker):
    """Training worker that performs neural network training.

    Samples batches from the Redis replay buffer and performs gradient
    updates, pushing new weights to Redis periodically.

    No Ray dependency - uses Redis for all coordination.

    Example:
        >>> worker = TrainingWorker(config={...})
        >>> worker.run(num_iterations=10000)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
    ):
        """Initialize the training worker.

        Args:
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
                - checkpoint_dir: Directory for saving checkpoints (default: 'checkpoints')
                - mlflow_tracking_uri: MLflow tracking server URI (optional)
                - mlflow_experiment_name: MLflow experiment name (optional)
                - bearoff_table_path: Path to bearoff table .npy file (optional)
                - use_bearoff_values: Whether to use perfect values for bearoff positions (default: True)
            worker_id: Optional unique worker ID. Auto-generated if not provided.
        """
        super().__init__(config, worker_id)

        # Training configuration
        self.train_batch_size = self.config.get('train_batch_size', 128)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.l2_reg_lambda = self.config.get('l2_reg_lambda', 1e-4)
        self.checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        self.min_buffer_size = self.config.get('min_buffer_size', 1000)
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')

        # Collection-gated training configuration
        # Training triggers after this many new games collected
        self.games_per_training_batch = self.config.get('games_per_training_batch', 10)
        # Number of training steps to run per collected game
        self.steps_per_game = self.config.get('steps_per_game', 10)

        # Surprise-weighted sampling configuration
        # 0 = uniform sampling, 1 = fully surprise-weighted
        self.surprise_weight = self.config.get('surprise_weight', 0.5)

        # Bearoff/endgame table configuration
        self.use_bearoff_values = self.config.get('use_bearoff_values', True)
        self._bearoff_table = None
        # Keep table on CPU (too large for GPU). Stored as [win, gammon_win, loss, gammon_loss].
        self._bearoff_table_np = None
        self._bearoff_lookup: Optional[BearoffLookup] = None
        bearoff_table_path = self.config.get('bearoff_table_path')
        if bearoff_table_path is None:
            # Default path
            default_path = Path(__file__).parent.parent.parent / 'data' / 'bearoff_15.npy'
            if default_path.exists():
                bearoff_table_path = str(default_path)

        if bearoff_table_path and self.use_bearoff_values:
            try:
                # Load table but DON'T convert to JAX array (too large for GPU)
                # We'll use numpy lookups instead. Supports legacy (n, n) win-prob
                # tables by expanding to the new (n, n, 4) format or packed upper.
                self._bearoff_lookup, self._bearoff_table_np = self._load_bearoff_table(bearoff_table_path)
                self._bearoff_table = True  # Flag indicating table is loaded
                shape_str = self._bearoff_table_np.shape
                size_gb = self._bearoff_table_np.nbytes / 1e9
                print(f"Worker {self.worker_id}: Loaded bearoff table from {bearoff_table_path}")
                print(f"Worker {self.worker_id}: Table shape: {shape_str}, size: {size_gb:.2f} GB")
            except Exception as e:
                print(f"Worker {self.worker_id}: Failed to load bearoff table: {e}")
                self._bearoff_table = None
                self._bearoff_table_np = None
                self._bearoff_lookup = None

        # Warm tree configuration
        # Number of MCTS simulations for warm tree (0 to disable)
        self.warm_tree_simulations = self.config.get('warm_tree_simulations', 0)
        self.warm_tree_max_nodes = self.config.get('warm_tree_max_nodes', 10000)

        # Warm tree components (lazy initialization)
        self._warm_tree_env = None
        self._warm_tree_evaluator = None

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

        # MLflow configuration
        self.mlflow_tracking_uri = self.config.get('mlflow_tracking_uri')
        self.mlflow_experiment_name = self.config.get('mlflow_experiment_name', 'bgai-training')
        self._mlflow_run = None

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

    # =========================================================================
    # Bearoff/Endgame Value Methods
    # =========================================================================

    def _load_bearoff_table(self, path: str) -> Tuple[BearoffLookup, np.ndarray]:
        """Load bearoff table in packed or full form.

        Normalizes to float32 and returns both the lookup wrapper and the
        underlying numpy array (kept for size logging).
        """
        table = np.load(path, mmap_mode='r')
        packed = False
        n = None

        if table.ndim == 1:
            # Packed flat (entries*4) not supported
            raise ValueError(f"Unexpected 1D bearoff table shape {table.shape}")

        if table.ndim == 2 and table.shape[1] in (4, 7) and table.shape[0] != table.shape[1]:
            # Packed upper: (entries, 4 or 7)
            packed = True
            n = solve_size_to_n(table.shape[0])
            table = table.astype(np.float32, copy=False)
            lookup = BearoffLookup(table, packed=True, n=n)
            return lookup, table

        if table.ndim == 2:
            # Legacy file storing P(win) only. Assume no gammons in the table.
            win = table.astype(np.float32, copy=False)
            loss = 1.0 - win
            zeros = np.zeros_like(win, dtype=np.float32)
            equity = (2.0 * win - 1.0)  # cubeless equity
            full = np.stack([win, zeros, loss, zeros, equity, equity, -equity], axis=-1)
            n = full.shape[0]
            lookup = BearoffLookup(full, packed=False, n=n)
            return lookup, full

        if table.ndim == 3 and table.shape[-1] in (4, 7):
            n = table.shape[0]
            lookup = BearoffLookup(table.astype(np.float32, copy=False), packed=False, n=n)
            return lookup, table

        raise ValueError(
            f"Unexpected bearoff table shape {table.shape}; expected packed (entries,4|7), legacy (n,n), or full (n,n,4|7)"
        )

    def _get_bearoff_value_np(self, board: np.ndarray, cur_player: int) -> float:
        """Get perfect bearoff equity from the table (numpy version).

        Equity is scaled to [-1, 1] using cubeless points (no cube).

        Args:
            board: Board array of shape (28,)
            cur_player: Current player (0 for X, 1 for O)

        Returns:
            Equity from current player's perspective.
        """
        if self._bearoff_table_np is None:
            return 0.0

        # Extract X's home board (points 0-5)
        x_pos = np.maximum(0, board[0:6])

        # Extract O's home board (points 18-23, stored as negative)
        # Flip the order so o_pos[0] = O's point closest to bear off
        o_pos = np.maximum(0, -board[23:17:-1])

        from bgai.endgame.indexing import position_to_index_fast

        x_idx = position_to_index_fast(x_pos)
        o_idx = position_to_index_fast(o_pos)

        probs = self._get_bearoff_prob_vector(x_idx, o_idx, cur_player)
        p_win = probs[0]
        p_gammon_win = probs[1]
        p_gammon_loss = probs[3]

        # Cubeless equity with gammons valued at 2 points
        equity = (2.0 * p_win - 1.0) + (p_gammon_win - p_gammon_loss)
        return float(equity)

    def _get_bearoff_prob_vector(self, x_idx: int, o_idx: int, cur_player: int) -> np.ndarray:
        """Return [win, gammon_win, loss, gammon_loss, (opt cube eq...)] for current player."""
        if self._bearoff_lookup is None:
            return np.zeros(4, dtype=np.float32)

        return self._bearoff_lookup.probs_for_player(x_idx, o_idx, cur_player)

    def _get_bearoff_target_probs(self, x_idx: int, o_idx: int, cur_player: int) -> np.ndarray:
        """Get 6-way probability distribution for bearoff position.

        Returns:
            [single win, gammon win, backgammon win,
             single loss, gammon loss, backgammon loss]
        """
        probs = self._get_bearoff_prob_vector(x_idx, o_idx, cur_player)
        win, gammon_win, loss, gammon_loss = probs
        return np.array([
            win - gammon_win,   # single win
            gammon_win,         # gammon win
            0.0,                # backgammon win (impossible in bearoff)
            loss - gammon_loss, # single loss
            gammon_loss,        # gammon loss
            0.0,                # backgammon loss (impossible)
        ], dtype=np.float32)

    def _apply_bearoff_values_to_batch(
        self,
        batch: BaseExperience,
    ) -> Tuple[BaseExperience, Dict[str, jnp.ndarray]]:
        """Replace rewards with perfect bearoff values where applicable.

        Uses numpy for bearoff detection and table lookup (table is too large for GPU),
        then converts results back to JAX arrays.

        Args:
            batch: Batch of experiences

        Returns:
            Tuple of (modified_batch, bearoff_metrics)
        """
        if self._bearoff_table_np is None or not self.use_bearoff_values:
            return batch, {'bearoff_count': jnp.array(0), 'total_count': jnp.array(batch.reward.shape[0])}

        batch_size = batch.observation_nn.shape[0]

        # Convert batch data to numpy for processing
        obs_np = np.array(batch.observation_nn)
        cur_player_np = np.array(batch.cur_player_id)
        rewards_np = np.array(batch.reward)
        from bgai.endgame.indexing import position_to_index_fast

        # Process each experience
        is_bearoff_list = []
        perfect_values_list = []
        target_probs_list = []

        for i in range(batch_size):
            # Decode observation to board
            board = np.round(obs_np[i, :28] * 15).astype(np.int8)

            # Check if bearoff position (numpy version)
            # Board layout: [0]=current bar, [1-24]=points 1-24, [25]=opponent bar
            # For bearoff: both bars empty, X only in home (points 1-6), O only in home (points 19-24)
            bar_empty = (board[0] == 0) and (board[25] == 0)
            # X (positive) outside home if pieces on points 7-24 (indices 7:25)
            x_outside = np.any(board[7:25] > 0)
            # O (negative) outside home if pieces on points 1-18 (indices 1:19)
            o_outside = np.any(board[1:19] < 0)
            is_bo = bar_empty and not x_outside and not o_outside

            is_bearoff_list.append(is_bo)

            if is_bo:
                # Get perfect value
                cur_player = int(cur_player_np[i])

                # Index positions for lookup
                x_pos = np.maximum(0, board[0:6])
                o_pos = np.maximum(0, -board[23:17:-1])
                x_idx = position_to_index_fast(x_pos)
                o_idx = position_to_index_fast(o_pos)

                perfect_val = self._get_bearoff_value_np(board, cur_player)
                perfect_values_list.append(perfect_val)
                target_probs_list.append(self._get_bearoff_target_probs(x_idx, o_idx, cur_player))
            else:
                perfect_values_list.append(0.0)
                target_probs_list.append(None)

        is_bearoff = np.array(is_bearoff_list)
        perfect_values = np.array(perfect_values_list)

        # Update rewards for bearoff positions
        new_rewards = rewards_np.copy()
        for i in range(batch_size):
            if is_bearoff[i]:
                cur_p = int(cur_player_np[i])
                opp_p = 1 - cur_p
                # Equity from current player's perspective; zero-sum for opponent
                new_rewards[i, cur_p] = perfect_values[i]
                new_rewards[i, opp_p] = -perfect_values[i]

        # Convert back to JAX and create new batch
        new_rewards_jax = jnp.array(new_rewards)
        modified_batch = batch.replace(reward=new_rewards_jax)

        # Metrics
        bearoff_count = int(np.sum(is_bearoff))
        metrics = {
            'bearoff_count': jnp.array(bearoff_count),
            'total_count': jnp.array(batch_size),
        }
        valid_targets = [tp for tp in target_probs_list if tp is not None]
        if valid_targets:
            stacked = np.stack(valid_targets)
            metrics['bearoff_avg_gammon_win'] = jnp.array(float(np.mean(stacked[:, 1])))

        return modified_batch, metrics

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking if configured."""
        if not self.mlflow_tracking_uri:
            print(f"Worker {self.worker_id}: MLflow not configured, skipping")
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)

            # Check if we should resume an existing run
            run_id = self.state.get_run_id()
            is_new_run = False

            if run_id:
                # Try to resume existing run
                try:
                    self._mlflow_run = mlflow.start_run(run_id=run_id)
                    print(f"Worker {self.worker_id}: Resumed MLflow run {run_id}")
                except Exception as resume_error:
                    # Run doesn't exist (e.g., MLFlow DB was reset), create new one
                    print(f"Worker {self.worker_id}: MLflow run {run_id} not found, creating new run")
                    self._mlflow_run = mlflow.start_run()
                    new_run_id = self._mlflow_run.info.run_id
                    self.state.set_run_id(new_run_id)
                    print(f"Worker {self.worker_id}: Started new MLflow run {new_run_id}")
                    run_id = new_run_id
                    is_new_run = True
            else:
                # Start new run
                self._mlflow_run = mlflow.start_run()
                run_id = self._mlflow_run.info.run_id
                self.state.set_run_id(run_id)
                print(f"Worker {self.worker_id}: Started MLflow run {run_id}")
                is_new_run = True

            # Log configuration parameters for new runs
            if is_new_run:
                mlflow.log_params({
                    'train_batch_size': self.train_batch_size,
                    'learning_rate': self.learning_rate,
                    'l2_reg_lambda': self.l2_reg_lambda,
                    'games_per_training_batch': self.games_per_training_batch,
                    'steps_per_game': self.steps_per_game,
                    'surprise_weight': self.surprise_weight,
                    'min_buffer_size': self.min_buffer_size,
                    'checkpoint_interval': self.checkpoint_interval,
                })

        except Exception as e:
            print(f"Worker {self.worker_id}: MLflow setup error: {e}")
            self._mlflow_run = None

    def _log_mlflow_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to MLflow if configured.

        Args:
            metrics: Dict of metric name to value.
            step: Training step number.
        """
        if self._mlflow_run is None:
            return

        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Worker {self.worker_id}: MLflow logging error: {e}")

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
        """Initialize training state from scratch or from Redis."""
        # Try to get weights from Redis
        result = self.get_current_model_weights()

        # Create sample input for initialization
        key = jax.random.PRNGKey(42)
        sample_state = self._env.init(key)
        sample_obs = sample_state.observation

        if result is not None:
            # Use weights from Redis
            weights_bytes, version = result
            params_dict = deserialize_weights(weights_bytes)
            params = params_dict['params']
            self.current_model_version = version
            print(f"Worker {self.worker_id}: Loaded model version {version}")
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

    def _push_weights_to_redis(self) -> bool:
        """Push current weights to Redis.

        Returns:
            True if weights were pushed successfully.
        """
        try:
            # Serialize weights
            params_dict = {'params': self._train_state.params}
            weights_bytes = serialize_weights(params_dict)

            # Increment version
            new_version = self.current_model_version + 1

            # Push to Redis
            success = self.state.set_model_weights(weights_bytes, new_version)

            if success:
                self.current_model_version = new_version
                print(f"Worker {self.worker_id}: Pushed weights version {new_version}")

                # Build and push warm tree if configured
                if self.warm_tree_simulations > 0:
                    self._build_and_push_warm_tree()

                return True

            # Version conflict - another worker updated
            current = self.state.get_model_version()
            print(f"Worker {self.worker_id}: Weight push rejected, current version is {current}")
            return False

        except Exception as e:
            print(f"Worker {self.worker_id}: Error pushing weights: {e}")
            return False

    def _setup_warm_tree_environment(self) -> None:
        """Set up environment for warm tree building (lazy initialization)."""
        if self._warm_tree_env is not None:
            return

        import pgx.backgammon as bg
        from core.types import StepMetadata

        self._warm_tree_env = bg.Backgammon(short_game=True)

        def step_fn(state, action, key):
            def stochastic_branch(operand):
                s, a, _ = operand
                return self._warm_tree_env.stochastic_step(s, a)

            def deterministic_branch(operand):
                s, a, k = operand
                return self._warm_tree_env.step(s, a, k)

            new_state = jax.lax.cond(
                state._is_stochastic,
                stochastic_branch,
                deterministic_branch,
                (state, action, key)
            )

            metadata = StepMetadata(
                rewards=new_state.rewards,
                action_mask=new_state.legal_action_mask,
                terminated=new_state.terminated,
                cur_player_id=new_state.current_player,
                step=new_state._step_count
            )

            return new_state, metadata

        self._warm_tree_step_fn = step_fn

        def init_fn(key):
            state = self._warm_tree_env.init(key)
            return state, StepMetadata(
                rewards=state.rewards,
                action_mask=state.legal_action_mask,
                terminated=state.terminated,
                cur_player_id=state.current_player,
                step=state._step_count
            )

        self._warm_tree_init_fn = init_fn
        self._warm_tree_state_to_nn = lambda state: state.observation

    def _setup_warm_tree_evaluator(self) -> None:
        """Set up MCTS evaluator for warm tree building."""
        if self._warm_tree_evaluator is not None:
            return

        self._setup_warm_tree_environment()

        from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
        from core.evaluators.mcts.action_selection import PUCTSelector
        from core.evaluators.evaluation_fns import make_nn_eval_fn

        eval_fn = make_nn_eval_fn(self._nn_model, self._warm_tree_state_to_nn)

        self._warm_tree_evaluator = StochasticMCTS(
            eval_fn=eval_fn,
            stochastic_action_probs=self._warm_tree_env.stochastic_action_probs,
            num_iterations=self.warm_tree_simulations,
            max_nodes=self.warm_tree_max_nodes,
            branching_factor=self._warm_tree_env.num_actions,
            action_selector=PUCTSelector(),
            temperature=1.0,
            persist_tree=True,
        )

    def _build_and_push_warm_tree(self) -> bool:
        """Build warm tree from initial position and push to Redis.

        Returns:
            True if warm tree was built and pushed successfully.
        """
        if self.warm_tree_simulations <= 0:
            return False

        try:
            start_time = time.time()
            print(f"Worker {self.worker_id}: Building warm tree ({self.warm_tree_simulations} sims)...")

            # Setup evaluator if needed
            self._setup_warm_tree_evaluator()

            # Initialize from starting position
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            init_state, init_metadata = self._warm_tree_init_fn(key)

            # Initialize tree
            eval_state = self._warm_tree_evaluator.init(template_embedding=init_state)

            # Get current params
            params = self._train_state.params

            # Run MCTS iterations to build the tree
            # The evaluator's evaluate() method runs num_iterations internally
            key, eval_key = jax.random.split(key)
            output = self._warm_tree_evaluator.evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=init_state,
                root_metadata=init_metadata,
                params=params,
                env_step_fn=self._warm_tree_step_fn,
            )

            # Serialize and push the warm tree
            warm_tree = output.eval_state
            tree_bytes = serialize_warm_tree(warm_tree)

            self.state.set_warm_tree(tree_bytes, self.current_model_version)

            duration = time.time() - start_time
            tree_size_mb = len(tree_bytes) / (1024 * 1024)
            print(
                f"Worker {self.worker_id}: Warm tree built and pushed "
                f"(v{self.current_model_version}, {tree_size_mb:.2f} MB, {duration:.1f}s)"
            )

            return True

        except Exception as e:
            print(f"Worker {self.worker_id}: Error building warm tree: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current training state."""
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

        # Log checkpoint to MLflow if configured
        if self._mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_artifact(ckpt_path)
            except Exception as e:
                print(f"Worker {self.worker_id}: MLflow artifact logging error: {e}")

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

        # Apply perfect bearoff values where applicable
        bearoff_metrics = None
        if self._bearoff_table is not None and self.use_bearoff_values:
            try:
                jax_batch, bearoff_metrics = self._apply_bearoff_values_to_batch(jax_batch)
            except Exception as e:
                print(f"Worker {self.worker_id}: Error applying bearoff values: {e}")
                import traceback
                traceback.print_exc()

        # Perform training step
        self._train_state, metrics = self._train_step(
            self._train_state,
            jax_batch,
        )

        # Convert metrics to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}

        # Add bearoff metrics
        if bearoff_metrics is not None:
            bearoff_count = int(bearoff_metrics['bearoff_count'])
            total_count = int(bearoff_metrics['total_count'])
            metrics['bearoff_count'] = bearoff_count
            metrics['bearoff_pct'] = 100.0 * bearoff_count / max(total_count, 1)
            if 'bearoff_avg_gammon_win' in bearoff_metrics:
                metrics['bearoff_avg_gammon_win'] = float(bearoff_metrics['bearoff_avg_gammon_win'])

            # Update training stats
            self._training_stats.bearoff_experiences += bearoff_count
            self._training_stats.non_bearoff_experiences += (total_count - bearoff_count)

        self._total_steps += 1
        self.stats.training_steps += 1

        return metrics

    def _get_current_games_count(self) -> int:
        """Get total number of completed games ever added (monotonic).

        Uses monotonic counter instead of buffer length to avoid stalling
        when buffer is full and old episodes are being evicted.
        """
        return self.buffer.get_total_games()

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

        # Set up MLflow tracking
        self._setup_mlflow()

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
            # Check if training is active
            if not self.is_training_active():
                print(f"Worker {self.worker_id}: Training paused, waiting...")
                time.sleep(5.0)
                continue

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

            # Push updated weights to Redis
            self._push_weights_to_redis()

            # Update tracking
            self._training_stats.games_at_last_train = current_games
            self._training_stats.experiences_at_last_train = self.buffer.get_size()

            # Log batch results
            elapsed = time.time() - start_time
            overall_steps_per_sec = self._total_steps / max(elapsed, 0.001)

            # Log to MLflow
            mlflow_metrics = {
                'loss': batch_metrics.get('loss', 0),
                'policy_loss': batch_metrics.get('policy_loss', 0),
                'value_loss': batch_metrics.get('value_loss', 0),
                'batch_steps': batch_metrics.get('batch_steps', 0),
                'batch_duration': batch_metrics.get('batch_duration', 0),
                'steps_per_sec': batch_metrics.get('steps_per_sec', 0),
                'overall_steps_per_sec': overall_steps_per_sec,
                'buffer_size': buffer_size,
                'total_games': current_games,
                'model_version': self.current_model_version,
            }

            # Add bearoff metrics if available
            if 'bearoff_pct' in batch_metrics:
                mlflow_metrics['bearoff_pct'] = batch_metrics.get('bearoff_pct', 0)
                mlflow_metrics['bearoff_count'] = batch_metrics.get('bearoff_count', 0)
                # Compute overall bearoff percentage
                total_bearoff = self._training_stats.bearoff_experiences
                total_non_bearoff = self._training_stats.non_bearoff_experiences
                total_exp = total_bearoff + total_non_bearoff
                if total_exp > 0:
                    mlflow_metrics['bearoff_pct_overall'] = 100.0 * total_bearoff / total_exp

            self._log_mlflow_metrics(mlflow_metrics, step=self._total_steps)

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

            # Build log message with optional bearoff stats
            log_msg = (
                f"Worker {self.worker_id}: Batch complete! "
                f"step={self._total_steps}, "
                f"loss={batch_metrics.get('loss', 0):.4f}, "
                f"batch_steps={batch_metrics.get('batch_steps', 0)}, "
                f"batch_time={batch_metrics.get('batch_duration', 0):.1f}s, "
                f"batch_steps/s={batch_metrics.get('steps_per_sec', 0):.1f}, "
                f"overall_steps/s={overall_steps_per_sec:.1f}, "
                f"version={self.current_model_version}"
            )
            if 'bearoff_pct' in batch_metrics:
                log_msg += f", bearoff={batch_metrics.get('bearoff_pct', 0):.1f}%"
            print(log_msg)

            last_log_time = time.time()

        # Final checkpoint
        if self._total_steps > 0:
            self._push_weights_to_redis()
            self._save_checkpoint()

        # Note: Don't call mlflow.end_run() here - the run is shared with eval workers
        # and should persist across worker restarts. The run will be ended manually
        # via the CLI or when the experiment is complete.
        self._mlflow_run = None

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

        # Compute bearoff percentage
        total_bearoff = self._training_stats.bearoff_experiences
        total_non_bearoff = self._training_stats.non_bearoff_experiences
        total_exp = total_bearoff + total_non_bearoff
        bearoff_pct = 100.0 * total_bearoff / max(total_exp, 1)

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
            # Bearoff/endgame stats
            'bearoff_experiences': total_bearoff,
            'non_bearoff_experiences': total_non_bearoff,
            'bearoff_pct': bearoff_pct,
            'bearoff_table_loaded': self._bearoff_table is not None,
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
