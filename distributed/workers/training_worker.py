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
from core.evaluators.mcts.equity import terminal_value_probs_from_reward, probs_to_equity

# Bearoff table for perfect endgame values (imports lazy-loaded in methods)


def weighted_az_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    experience: BaseExperience,
    sample_weights: jnp.ndarray,
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """AlphaZero loss function with per-sample weighting and 6-way value head.

    This allows us to weight bearoff samples more heavily since they have
    known-perfect target values. Value loss is cross-entropy over 6 outcomes:
    [win, gammon_win, backgammon_win, loss, gammon_loss, backgammon_loss]

    Args:
        params: Neural network parameters
        train_state: Flax TrainState
        experience: Batch of experiences
        sample_weights: Per-sample weights (e.g., higher for bearoff positions)
        l2_reg_lambda: L2 regularization weight

    Returns:
        (loss, (aux_metrics, updates))
    """
    # Get batch_stats if using batch_norm
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    # Get predictions - value head now outputs 6-way logits
    (pred_policy_logits, pred_value_logits), updates = train_state.apply_fn(
        variables,
        x=experience.observation_nn,
        mutable=mutables
    )

    # Set invalid actions in policy to -inf
    pred_policy_logits = jnp.where(
        experience.policy_mask,
        pred_policy_logits,
        jnp.finfo(jnp.float32).min
    )

    # Compute per-sample policy loss (cross entropy)
    policy_loss_per_sample = optax.softmax_cross_entropy(pred_policy_logits, experience.policy_weights)
    # Apply weights and average
    policy_loss = jnp.sum(policy_loss_per_sample * sample_weights) / jnp.sum(sample_weights)

    # Select appropriate reward from experience.reward for current player
    current_player = experience.cur_player_id
    batch_indices = jnp.arange(experience.reward.shape[0])
    target_reward = experience.reward[batch_indices, current_player]

    # Convert scalar reward to 6-way target probability distribution
    target_value_probs = terminal_value_probs_from_reward(target_reward)

    # Compute per-sample value loss (cross-entropy over 6 outcomes)
    pred_value_log_probs = jax.nn.log_softmax(pred_value_logits, axis=-1)
    value_loss_per_sample = -(target_value_probs * pred_value_log_probs).sum(axis=-1)
    # Apply weights and average
    value_loss = jnp.sum(value_loss_per_sample * sample_weights) / jnp.sum(sample_weights)

    # Compute L2 regularization (not weighted by samples)
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda x: (x ** 2).sum(), params)
    )

    # Total loss
    loss = policy_loss + value_loss + l2_reg

    # Compute metrics (unweighted for interpretability)
    pred_policy_probs = jax.nn.softmax(pred_policy_logits, axis=-1)
    pred_top1 = jnp.argmax(pred_policy_probs, axis=-1)
    target_top1 = jnp.argmax(experience.policy_weights, axis=-1)
    policy_accuracy = jnp.mean(pred_top1 == target_top1)

    # Value accuracy: top-1 match between predicted and target outcome
    pred_value_probs = jax.nn.softmax(pred_value_logits, axis=-1)
    value_top1 = jnp.argmax(pred_value_probs, axis=-1)
    target_value_top1 = jnp.argmax(target_value_probs, axis=-1)
    value_accuracy = jnp.mean(value_top1 == target_value_top1)

    # Per-outcome value losses for monitoring (6-way breakdown)
    # Compute per-outcome cross-entropy contribution: -target_prob * log(pred_prob)
    # This shows which outcomes are hardest to predict
    per_outcome_ce = -target_value_probs * pred_value_log_probs  # (batch, 6)
    mean_per_outcome_ce = per_outcome_ce.mean(axis=0)  # (6,)

    # Outcome names: win, gammon_win, backgammon_win, loss, gammon_loss, backgammon_loss
    value_loss_win = mean_per_outcome_ce[0]
    value_loss_gammon_win = mean_per_outcome_ce[1]
    value_loss_backgammon_win = mean_per_outcome_ce[2]
    value_loss_loss = mean_per_outcome_ce[3]
    value_loss_gammon_loss = mean_per_outcome_ce[4]
    value_loss_backgammon_loss = mean_per_outcome_ce[5]

    # Predicted outcome distribution (mean over batch)
    pred_value_probs = jax.nn.softmax(pred_value_logits, axis=-1)
    mean_pred_probs = pred_value_probs.mean(axis=0)  # (6,)

    # Target outcome distribution (mean over batch)
    mean_target_probs = target_value_probs.mean(axis=0)  # (6,)

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        'policy_accuracy': policy_accuracy,
        'value_accuracy': value_accuracy,
        # Per-outcome value losses
        'value_loss_win': value_loss_win,
        'value_loss_gammon_win': value_loss_gammon_win,
        'value_loss_backgammon_win': value_loss_backgammon_win,
        'value_loss_loss': value_loss_loss,
        'value_loss_gammon_loss': value_loss_gammon_loss,
        'value_loss_backgammon_loss': value_loss_backgammon_loss,
        # Predicted outcome probabilities (for calibration monitoring)
        'pred_prob_win': mean_pred_probs[0],
        'pred_prob_gammon_win': mean_pred_probs[1],
        'pred_prob_backgammon_win': mean_pred_probs[2],
        'pred_prob_loss': mean_pred_probs[3],
        'pred_prob_gammon_loss': mean_pred_probs[4],
        'pred_prob_backgammon_loss': mean_pred_probs[5],
        # Target outcome probabilities (ground truth distribution)
        'target_prob_win': mean_target_probs[0],
        'target_prob_gammon_win': mean_target_probs[1],
        'target_prob_backgammon_win': mean_target_probs[2],
        'target_prob_loss': mean_target_probs[3],
        'target_prob_gammon_loss': mean_target_probs[4],
        'target_prob_backgammon_loss': mean_target_probs[5],
    }
    return loss, (aux_metrics, updates)

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
    bearoff_value_loss_count: int = 0
    non_bearoff_value_loss_count: int = 0
    last_batch_duration: float = 0.0
    last_batch_steps: int = 0
    cumulative_loss: float = 0.0
    loss_count: int = 0

    @property
    def avg_loss(self) -> float:
        return self.cumulative_loss / max(self.loss_count, 1)

    @property
    def avg_bearoff_value_loss(self) -> float:
        return self.cumulative_bearoff_value_loss / max(self.bearoff_value_loss_count, 1)

    @property
    def avg_non_bearoff_value_loss(self) -> float:
        return self.cumulative_non_bearoff_value_loss / max(self.non_bearoff_value_loss_count, 1)


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
                - games_per_epoch: New games to trigger training epoch (default: 10)
                - checkpoint_epoch_interval: Epochs between checkpoints (default: 5)
                - max_checkpoints: Maximum checkpoints to keep (default: 5)
                - min_buffer_size: Minimum buffer size before training (default: 1000)
                - redis_host: Redis server host (default: 'localhost')
                - redis_port: Redis server port (default: 6379)
                - checkpoint_dir: Directory for saving checkpoints (default: 'checkpoints')
                - mlflow_tracking_uri: MLflow tracking server URI (optional)
                - mlflow_experiment_name: MLflow experiment name (optional)
                - bearoff_enabled: Whether bearoff DB is enabled (default: False)
                - bearoff_value_weight: Learning weight for bearoff positions (default: 2.0)
                - lookup_enabled: Whether lookup positions are enabled (default: False)
                - lookup_learning_weight: Learning weight for lookup positions (default: 1.5)
                - network_hidden_dim: Network hidden dimension (default: 256)
                - network_num_blocks: Number of residual blocks (default: 6)
            worker_id: Optional unique worker ID. Auto-generated if not provided.
        """
        super().__init__(config, worker_id)

        # Training configuration
        self.train_batch_size = self.config.get('train_batch_size', 128)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.l2_reg_lambda = self.config.get('l2_reg_lambda', 1e-4)
        # checkpoint_epoch_interval (new) with fallback to checkpoint_interval (old)
        self.checkpoint_epoch_interval = self.config.get(
            'checkpoint_epoch_interval',
            self.config.get('checkpoint_interval', 5)
        )
        self.max_checkpoints = self.config.get('max_checkpoints', 5)
        self.min_buffer_size = self.config.get('min_buffer_size', 1000)
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')

        # Collection-gated training configuration
        # Training triggers after this many new games collected (epoch = games_per_epoch games)
        # games_per_epoch (new) with fallback to games_per_training_batch (old)
        self.games_per_epoch = self.config.get(
            'games_per_epoch',
            self.config.get('games_per_training_batch', 10)
        )
        # Number of training steps to run per collected game
        # Default high to train on most experiences (avg game ~60 moves / batch_size)
        # Set to 0 to auto-calculate based on buffer size
        self.steps_per_game = self.config.get('steps_per_game', 50)

        # Surprise-weighted sampling configuration
        # 0 = uniform sampling, 1 = fully surprise-weighted
        self.surprise_weight = self.config.get('surprise_weight', 0.5)

        # Bearoff/endgame table configuration
        self.bearoff_enabled = self.config.get('bearoff_enabled', False)
        # Weight multiplier for bearoff value loss (since bearoff values are known-perfect)
        # Higher values emphasize learning accurate values for endgame positions
        self.bearoff_value_weight = self.config.get('bearoff_value_weight', 2.0)
        # Lookup position configuration
        self.lookup_enabled = self.config.get('lookup_enabled', False)
        self.lookup_learning_weight = self.config.get('lookup_learning_weight', 1.5)
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

        if bearoff_table_path and self.bearoff_enabled:
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

        OPTIMIZED: Fully vectorized - no Python for-loops over batch.

        Args:
            batch: Batch of experiences

        Returns:
            Tuple of (modified_batch, bearoff_metrics)
            bearoff_metrics includes:
                - bearoff_count: number of bearoff positions
                - total_count: total batch size
                - is_bearoff_mask: JAX array boolean mask (True for bearoff positions)
        """
        if self._bearoff_table_np is None or not self.bearoff_enabled:
            batch_size = batch.reward.shape[0]
            return batch, {
                'bearoff_count': jnp.array(0),
                'total_count': jnp.array(batch_size),
                'is_bearoff_mask': jnp.zeros(batch_size, dtype=jnp.bool_),
            }

        batch_size = batch.observation_nn.shape[0]

        # Convert batch data to numpy for processing
        obs_np = np.array(batch.observation_nn)
        cur_player_np = np.array(batch.cur_player_id)
        rewards_np = np.array(batch.reward)
        from bgai.endgame.indexing import position_to_index_fast

        # Decode observations to boards (batch_size, 28)
        boards = np.round(obs_np[:, :28] * 15).astype(np.int8)

        # Process each experience for target probs (uses BearoffLookup)
        target_probs_list = []

        # =========================================================================
        # Vectorized bearoff detection
        # =========================================================================
        # Board layout (pgx): [0-23]=points 0-23, [24]=cur_bar, [25]=opp_bar, [26]=cur_borne, [27]=opp_borne
        # Current player pieces are positive, opponent pieces are negative
        # Current player home = points 18-23, opponent home = points 0-5
        # For bearoff: both bars empty, cur player only in home (18-23), opp only in home (0-5)

        # Check bars empty (vectorized)
        bar_empty = (boards[:, 24] == 0) & (boards[:, 25] == 0)

        # Current player (positive) outside home if pieces on points 0-17
        cur_outside = np.any(boards[:, 0:18] > 0, axis=1)

        # Opponent (negative) outside home if pieces on points 6-23
        opp_outside = np.any(boards[:, 6:24] < 0, axis=1)

        # Bearoff position mask
        is_bearoff = bar_empty & ~cur_outside & ~opp_outside

        # =========================================================================
        # Bearoff value lookup using BearoffLookup
        # =========================================================================
        bearoff_count = int(np.sum(is_bearoff))

        # Collect target probs for bearoff positions
        for i in range(batch_size):
            if is_bearoff[i]:
                board = boards[i]
                cur_player = int(cur_player_np[i])

                # Index positions for lookup
                x_pos = np.maximum(0, board[0:6])
                o_pos = np.maximum(0, -board[23:17:-1])
                x_idx = position_to_index_fast(x_pos)
                o_idx = position_to_index_fast(o_pos)

                target_probs_list.append(self._get_bearoff_target_probs(x_idx, o_idx, cur_player))
            else:
                target_probs_list.append(None)

        # Update rewards for bearoff positions
        new_rewards = rewards_np.copy()
        for i in range(batch_size):
            if is_bearoff[i]:
                cur_p = int(cur_player_np[i])
                opp_p = 1 - cur_p
                board = boards[i]
                # Get equity from bearoff table
                perfect_val = self._get_bearoff_value_np(board, cur_p)
                # Equity from current player's perspective; zero-sum for opponent
                new_rewards[i, cur_p] = perfect_val
                new_rewards[i, opp_p] = -perfect_val

        # Convert back to JAX and create new batch
        new_rewards_jax = jnp.array(new_rewards)
        modified_batch = batch.replace(reward=new_rewards_jax)

        # Metrics - include mask for separate loss tracking
        is_bearoff_jax = jnp.array(is_bearoff)
        metrics = {
            'bearoff_count': jnp.array(bearoff_count),
            'total_count': jnp.array(batch_size),
            'is_bearoff_mask': is_bearoff_jax,
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
                    self._mlflow_run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
                    print(f"Worker {self.worker_id}: Resumed MLflow run {run_id} (system metrics enabled)")
                except Exception as resume_error:
                    # Run doesn't exist (e.g., MLFlow DB was reset), create new one
                    print(f"Worker {self.worker_id}: MLflow run {run_id} not found, creating new run")
                    self._mlflow_run = mlflow.start_run(log_system_metrics=True)
                    new_run_id = self._mlflow_run.info.run_id
                    self.state.set_run_id(new_run_id)
                    print(f"Worker {self.worker_id}: Started new MLflow run {new_run_id} (system metrics enabled)")
                    run_id = new_run_id
                    is_new_run = True
            else:
                # Start new run
                self._mlflow_run = mlflow.start_run(log_system_metrics=True)
                run_id = self._mlflow_run.info.run_id
                self.state.set_run_id(run_id)
                print(f"Worker {self.worker_id}: Started MLflow run {run_id} (system metrics enabled)")
                is_new_run = True

            # Log ALL configuration parameters for new runs
            # MLflow allows comparing params between runs in the UI
            if is_new_run:
                # Training params
                mlflow.log_params({
                    'train_batch_size': self.train_batch_size,
                    'learning_rate': self.learning_rate,
                    'l2_reg_lambda': self.l2_reg_lambda,
                    'games_per_epoch': self.games_per_epoch,
                    'surprise_weight': self.surprise_weight,
                    'min_buffer_size': self.min_buffer_size,
                    'checkpoint_epoch_interval': self.checkpoint_epoch_interval,
                    'max_checkpoints': self.max_checkpoints,
                    # Bearoff/endgame params
                    'bearoff_enabled': self.bearoff_enabled,
                    'bearoff_value_weight': self.bearoff_value_weight,
                    'lookup_enabled': self.lookup_enabled,
                    'lookup_learning_weight': self.lookup_learning_weight,
                    # Warm tree params (now from mcts section)
                    'warm_tree_simulations': self.warm_tree_simulations,
                    'warm_tree_max_nodes': self.warm_tree_max_nodes,
                })
                # Log MCTS and game params from config (may not be set on training worker)
                mcts_config = self.config.get('mcts', {})
                game_config = self.config.get('game', {})
                network_config = self.config.get('network', {})
                redis_config = self.config.get('redis', {})
                gnubg_config = self.config.get('gnubg', {})
                mlflow.log_params({
                    # MCTS params (used by game workers)
                    'mcts_collect_simulations': mcts_config.get('collect_simulations', mcts_config.get('simulations', 100)),
                    'mcts_eval_simulations': mcts_config.get('eval_simulations', 50),
                    'mcts_max_nodes': mcts_config.get('max_nodes', 400),
                    'mcts_persist_tree': mcts_config.get('persist_tree', True),
                    # Temperature schedule (now in mcts section)
                    'mcts_temperature_start': mcts_config.get('temperature_start', 0.8),
                    'mcts_temperature_end': mcts_config.get('temperature_end', 0.2),
                    # Game params
                    'max_episode_steps': game_config.get('max_episode_steps', 500),
                    'short_game': game_config.get('short_game', True),
                    'simple_doubles': game_config.get('simple_doubles', False),
                    # Network architecture
                    'network_hidden_dim': network_config.get('hidden_dim', 256),
                    'network_num_blocks': network_config.get('num_blocks', 6),
                })
                # Redis buffer settings (affects data retention)
                mlflow.log_params({
                    'buffer_capacity': redis_config.get('buffer_capacity', 100000),
                    'episode_capacity': redis_config.get('episode_capacity', 5000),
                })
                # GNUBG evaluation settings (for eval workers)
                mlflow.log_params({
                    'gnubg_ply': gnubg_config.get('ply', 2),
                    'gnubg_shortcuts': gnubg_config.get('shortcuts', 0),
                    'gnubg_osdb': gnubg_config.get('osdb', 1),
                    'gnubg_move_filters': str(gnubg_config.get('move_filters', [8, 4, 2, 2])),
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
            """ResNet-style network with 6-way value head for backgammon outcomes.

            Value head outputs logits for 6 outcomes:
            [win, gammon_win, backgammon_win, loss, gammon_loss, backgammon_loss]
            """
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6
            value_head_out_size: int = 6  # 6-way outcome distribution

            @nn.compact
            def __call__(self, x, train: bool = False):  # noqa: ARG002 - train required by interface
                del train  # unused but required by turbozero interface
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                # 6-way value head: outputs logits, converted to probs by loss fn
                value_logits = nn.Dense(self.value_head_out_size)(x)
                return policy_logits, value_logits

        # Use network config from YAML, with fallback to defaults
        num_hidden = self.config.get('network_hidden_dim', 256)
        num_blocks = self.config.get('network_num_blocks', 6)
        self._nn_model = ResNetTurboZero(
            self._env.num_actions,
            num_hidden=num_hidden,
            num_blocks=num_blocks
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
            variables = self._nn_model.init(key, sample_obs[None, ...])
            params = variables['params']
            print(f"Worker {self.worker_id}: Initialized random weights")

        # Restore total steps from Redis (persisted across restarts)
        self._total_steps = self.state.get_training_steps()
        if self._total_steps > 0:
            print(f"Worker {self.worker_id}: Restored step counter to {self._total_steps}")

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

            # Get current params - wrap in {'params': ...} for Flax module apply
            params = {'params': self._train_state.params}

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
        """Perform a single training step (uniform weights).

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

        # Update batch_stats if present (for BatchNorm layers)
        if updates and 'batch_stats' in updates and hasattr(new_state, 'batch_stats'):
            new_state = new_state.replace(batch_stats=updates['batch_stats'])

        # Add loss to metrics
        metrics = {**metrics, 'loss': loss}

        return new_state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _train_step_weighted(
        self,
        train_state: TrainState,
        batch: BaseExperience,
        sample_weights: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Perform a single training step with per-sample weights.

        This enables weighting bearoff positions more heavily since they
        have known-perfect target values.

        Args:
            train_state: Current training state.
            batch: Batch of experiences.
            sample_weights: Per-sample weights (e.g., bearoff_value_weight for bearoff positions).

        Returns:
            Tuple of (updated_train_state, metrics_dict).
        """
        loss_fn = partial(weighted_az_loss_fn, l2_reg_lambda=self.l2_reg_lambda)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, updates)), grads = grad_fn(
            train_state.params,
            train_state,
            batch,
            sample_weights
        )

        # Apply gradients
        new_state = train_state.apply_gradients(grads=grads)

        # Update batch_stats if present (for BatchNorm layers)
        if updates and 'batch_stats' in updates and hasattr(new_state, 'batch_stats'):
            new_state = new_state.replace(batch_stats=updates['batch_stats'])

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
        sample_weights = None
        if self._bearoff_table is not None and self.bearoff_enabled:
            try:
                jax_batch, bearoff_metrics = self._apply_bearoff_values_to_batch(jax_batch)
                # Create sample weights: bearoff_value_weight for bearoff, 1.0 for others
                is_bearoff_mask = bearoff_metrics['is_bearoff_mask']
                sample_weights = jnp.where(
                    is_bearoff_mask,
                    self.bearoff_value_weight,
                    1.0
                )
            except Exception as e:
                print(f"Worker {self.worker_id}: Error applying bearoff values: {e}")
                import traceback
                traceback.print_exc()

        # Perform training step - use weighted version if we have weights
        if sample_weights is not None and self.bearoff_value_weight != 1.0:
            self._train_state, metrics = self._train_step_weighted(
                self._train_state,
                jax_batch,
                sample_weights,
            )
        else:
            self._train_state, metrics = self._train_step(
                self._train_state,
                jax_batch,
            )

        # Convert metrics to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}

        # Add bearoff metrics and compute separate value losses
        if bearoff_metrics is not None:
            bearoff_count = int(bearoff_metrics['bearoff_count'])
            total_count = int(bearoff_metrics['total_count'])
            metrics['bearoff_count'] = bearoff_count
            metrics['bearoff_pct'] = 100.0 * bearoff_count / max(total_count, 1)
            if 'bearoff_avg_gammon_win' in bearoff_metrics:
                metrics['bearoff_avg_gammon_win'] = float(bearoff_metrics['bearoff_avg_gammon_win'])

            # Compute separate losses for bearoff vs non-bearoff positions
            is_bearoff_mask = bearoff_metrics['is_bearoff_mask']
            if bearoff_count > 0:
                # Get predictions for the batch (no grad computation)
                pred_policy_logits, pred_value_logits = self._train_state.apply_fn(
                    {'params': self._train_state.params},
                    x=jax_batch.observation_nn,
                )

                # Mask invalid actions in policy for softmax
                pred_policy_logits = jnp.where(
                    jax_batch.policy_mask,
                    pred_policy_logits,
                    jnp.finfo(jnp.float32).min
                )

                # Get target values and convert to 6-way probabilities
                current_player = jax_batch.cur_player_id
                target_reward = jax_batch.reward[jnp.arange(jax_batch.reward.shape[0]), current_player]
                target_value_probs = terminal_value_probs_from_reward(target_reward)

                # Compute per-sample value cross-entropy
                pred_value_log_probs = jax.nn.log_softmax(pred_value_logits, axis=-1)
                value_ce = -(target_value_probs * pred_value_log_probs).sum(axis=-1)

                # Compute per-sample policy cross-entropy
                policy_ce = optax.softmax_cross_entropy(pred_policy_logits, jax_batch.policy_weights)

                bearoff_mask_float = is_bearoff_mask.astype(jnp.float32)
                non_bearoff_count = total_count - bearoff_count
                non_bearoff_mask_float = (1.0 - bearoff_mask_float)

                # Bearoff value loss (cross-entropy for bearoff positions only)
                bearoff_value_loss = jnp.sum(value_ce * bearoff_mask_float) / max(bearoff_count, 1)
                metrics['bearoff_value_loss'] = float(bearoff_value_loss)

                # Bearoff policy loss (cross-entropy for bearoff positions only)
                bearoff_policy_loss = jnp.sum(policy_ce * bearoff_mask_float) / max(bearoff_count, 1)
                metrics['bearoff_policy_loss'] = float(bearoff_policy_loss)

                # Non-bearoff value loss
                if non_bearoff_count > 0:
                    non_bearoff_value_loss = jnp.sum(value_ce * non_bearoff_mask_float) / non_bearoff_count
                    metrics['non_bearoff_value_loss'] = float(non_bearoff_value_loss)
                    # Non-bearoff policy loss
                    non_bearoff_policy_loss = jnp.sum(policy_ce * non_bearoff_mask_float) / non_bearoff_count
                    metrics['non_bearoff_policy_loss'] = float(non_bearoff_policy_loss)
                else:
                    metrics['non_bearoff_value_loss'] = 0.0
                    metrics['non_bearoff_policy_loss'] = 0.0

                # Update cumulative stats
                self._training_stats.cumulative_bearoff_value_loss += float(bearoff_value_loss) * bearoff_count
                self._training_stats.bearoff_value_loss_count += bearoff_count
            else:
                # No bearoff positions - all are non-bearoff
                metrics['bearoff_value_loss'] = 0.0
                metrics['bearoff_policy_loss'] = 0.0
                metrics['non_bearoff_value_loss'] = metrics.get('value_loss', 0.0)
                metrics['non_bearoff_policy_loss'] = metrics.get('policy_loss', 0.0)

            # Update experience counts
            self._training_stats.bearoff_experiences += bearoff_count
            non_bearoff_count = total_count - bearoff_count
            self._training_stats.non_bearoff_experiences += non_bearoff_count
            if non_bearoff_count > 0:
                self._training_stats.cumulative_non_bearoff_value_loss += metrics.get('value_loss', 0.0) * non_bearoff_count
                self._training_stats.non_bearoff_value_loss_count += non_bearoff_count

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

            # Persist step counter to Redis periodically for recovery
            if self._total_steps % 100 == 0:
                self.state.set_training_steps(self._total_steps)

        batch_duration = time.time() - batch_start

        # Update stats
        self._training_stats.last_batch_duration = batch_duration
        self._training_stats.last_batch_steps = steps_done
        self._training_stats.total_batches_trained += 1

        # Save checkpoint every N epochs (training batches)
        if self._training_stats.total_batches_trained % self.checkpoint_epoch_interval == 0:
            self._save_checkpoint()
            self.state.set_training_steps(self._total_steps)

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
        metrics_port_config = self.config.get('metrics_port', 9200)
        metrics_port = start_metrics_server(metrics_port_config)
        if metrics_port is None:
            print(f"Worker {self.worker_id}: Failed to start metrics server")
            metrics_port = metrics_port_config  # Fallback for registration
        metrics = get_metrics()

        # Register metrics endpoint for dynamic discovery (use actual bound port)
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
              f"for every {self.games_per_epoch} games collected")

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

            if new_games < self.games_per_epoch:
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
                        f"({new_games}/{self.games_per_epoch} new games), "
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
                f"Worker {self.worker_id}: Training epoch triggered! "
                f"{new_games} new games -> {target_steps} training steps"
            )

            # Run training epoch (game collection continues in parallel)
            batch_metrics = self._run_training_batch(target_steps)

            # Push updated weights to Redis
            self._push_weights_to_redis()

            # Update tracking
            self._training_stats.games_at_last_train = current_games
            self._training_stats.experiences_at_last_train = self.buffer.get_size()

            # Log batch results
            elapsed = time.time() - start_time
            overall_steps_per_sec = self._total_steps / max(elapsed, 0.001)

            # Log to MLflow - ordered by importance for ML dashboard
            # Priority: overall loss > value/policy > bearoff losses

            # Compute overall loss (sum of value + policy, unweighted for interpretability)
            value_loss = batch_metrics.get('non_bearoff_value_loss', batch_metrics.get('value_loss', 0))
            policy_loss = batch_metrics.get('non_bearoff_policy_loss', batch_metrics.get('policy_loss', 0))
            overall_loss = value_loss + policy_loss

            mlflow_metrics = {
                # 1. Overall loss - single number to track training progress
                'loss': overall_loss,
                # 2. Core losses (non-bearoff = main game positions)
                'value_loss': value_loss,
                'policy_loss': policy_loss,
                # 3. Bearoff-specific losses (endgame positions with known-perfect values)
                'bearoff_value_loss': batch_metrics.get('bearoff_value_loss', 0),
                'bearoff_policy_loss': batch_metrics.get('bearoff_policy_loss', 0),
                # 4. Per-outcome value losses (6-way breakdown)
                'value_loss_win': batch_metrics.get('value_loss_win', 0),
                'value_loss_gammon_win': batch_metrics.get('value_loss_gammon_win', 0),
                'value_loss_backgammon_win': batch_metrics.get('value_loss_backgammon_win', 0),
                'value_loss_loss': batch_metrics.get('value_loss_loss', 0),
                'value_loss_gammon_loss': batch_metrics.get('value_loss_gammon_loss', 0),
                'value_loss_backgammon_loss': batch_metrics.get('value_loss_backgammon_loss', 0),
                # 5. Predicted vs target outcome distributions (calibration)
                'pred_prob_win': batch_metrics.get('pred_prob_win', 0),
                'pred_prob_gammon_win': batch_metrics.get('pred_prob_gammon_win', 0),
                'pred_prob_backgammon_win': batch_metrics.get('pred_prob_backgammon_win', 0),
                'pred_prob_loss': batch_metrics.get('pred_prob_loss', 0),
                'pred_prob_gammon_loss': batch_metrics.get('pred_prob_gammon_loss', 0),
                'pred_prob_backgammon_loss': batch_metrics.get('pred_prob_backgammon_loss', 0),
                'target_prob_win': batch_metrics.get('target_prob_win', 0),
                'target_prob_gammon_win': batch_metrics.get('target_prob_gammon_win', 0),
                'target_prob_backgammon_win': batch_metrics.get('target_prob_backgammon_win', 0),
                'target_prob_loss': batch_metrics.get('target_prob_loss', 0),
                'target_prob_gammon_loss': batch_metrics.get('target_prob_gammon_loss', 0),
                'target_prob_backgammon_loss': batch_metrics.get('target_prob_backgammon_loss', 0),
                # 6. Accuracy metrics
                'value_accuracy': batch_metrics.get('value_accuracy', 0),
                'policy_accuracy': batch_metrics.get('policy_accuracy', 0),
                # 7. Training context
                'bearoff_train_pct': batch_metrics.get('bearoff_pct', 0),
                'train_steps_per_sec': batch_metrics.get('steps_per_sec', 0),
                'total_train_steps': self._total_steps,
                'buffer_size': buffer_size,
                'total_games': current_games,
                'games_this_batch': new_games,
                'model_version': self.current_model_version,
            }

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
            metrics.training_loss.labels(
                worker_id=self.worker_id, loss_type='value'
            ).set(batch_metrics.get('value_loss', 0))
            metrics.training_loss.labels(
                worker_id=self.worker_id, loss_type='policy'
            ).set(batch_metrics.get('policy_loss', 0))

            # Per-outcome value losses (6-way breakdown)
            outcome_names = ['win', 'gammon_win', 'backgammon_win', 'loss', 'gammon_loss', 'backgammon_loss']
            for outcome in outcome_names:
                metrics.value_loss_per_outcome.labels(
                    worker_id=self.worker_id, outcome=outcome
                ).set(batch_metrics.get(f'value_loss_{outcome}', 0))
                metrics.predicted_outcome_prob.labels(
                    worker_id=self.worker_id, outcome=outcome
                ).set(batch_metrics.get(f'pred_prob_{outcome}', 0))
                metrics.target_outcome_prob.labels(
                    worker_id=self.worker_id, outcome=outcome
                ).set(batch_metrics.get(f'target_prob_{outcome}', 0))

            # Accuracy metrics
            metrics.value_accuracy.labels(
                worker_id=self.worker_id
            ).set(batch_metrics.get('value_accuracy', 0))
            metrics.policy_accuracy.labels(
                worker_id=self.worker_id
            ).set(batch_metrics.get('policy_accuracy', 0))

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
                f"Worker {self.worker_id}: Epoch complete! "
                f"step={self._total_steps}, "
                f"loss={batch_metrics.get('loss', 0):.4f}, "
                f"epoch_steps={batch_metrics.get('batch_steps', 0)}, "
                f"epoch_time={batch_metrics.get('batch_duration', 0):.1f}s, "
                f"train_steps/s={batch_metrics.get('steps_per_sec', 0):.1f}, "
                f"overall_steps/s={overall_steps_per_sec:.1f}, "
                f"version={self.current_model_version}"
            )
            if 'bearoff_pct' in batch_metrics:
                bearoff_pct = batch_metrics.get('bearoff_pct', 0)
                log_msg += f", bearoff={bearoff_pct:.1f}%"
                # Show separate value losses when we have bearoff positions
                if bearoff_pct > 0:
                    log_msg += f" (bo_vloss={batch_metrics.get('bearoff_value_loss', 0):.4f}, other_vloss={batch_metrics.get('non_bearoff_value_loss', 0):.4f})"
            print(log_msg)

            last_log_time = time.time()

        # Final checkpoint
        if self._total_steps > 0:
            self._push_weights_to_redis()
            self._save_checkpoint()
            # Persist step counter to Redis for recovery
            self.state.set_training_steps(self._total_steps)

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
            'games_per_training_batch': self.games_per_epoch,
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
