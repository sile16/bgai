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
import threading
import queue
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
from bgai.endgame.packed_table import V4BearoffLookup, load_v4_bearoff
from core.evaluators.mcts.equity import value_outputs_to_equity

# Bearoff table for perfect endgame values (imports lazy-loaded in methods)
# 4-way value/equity helpers are imported from TurboZero.


def weighted_az_loss_fn(
    params: chex.ArrayTree,
    train_state: TrainState,
    experience: BaseExperience,
    value_targets: jnp.ndarray,
    value_masks: jnp.ndarray,
    value_sample_weights: jnp.ndarray,
    policy_sample_weights: jnp.ndarray,
    is_bearoff_mask: jnp.ndarray,
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """AlphaZero loss function with per-sample weighting and 4-way conditional value head.

    This allows us to weight bearoff value samples more heavily without also
    overweighting policy loss. Value head outputs 4 independent sigmoids:
    [win, gam_win_cond, gam_loss_cond, bg_rate]

    Args:
        params: Neural network parameters
        train_state: Flax TrainState
        experience: Batch of experiences
        value_targets: Precomputed 4-way value targets
        value_masks: Masks for which targets are valid
        value_sample_weights: Per-sample weights for value loss (e.g., higher for bearoff)
        policy_sample_weights: Per-sample weights for policy loss
        is_bearoff_mask: Boolean mask indicating bearoff positions
        l2_reg_lambda: L2 regularization weight

    Returns:
        (loss, (aux_metrics, updates))
    """
    # Get batch_stats if using batch_norm
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    # Get predictions - value head outputs 4-way logits (sigmoid, not softmax)
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
    policy_loss = jnp.sum(policy_loss_per_sample * policy_sample_weights) / jnp.maximum(jnp.sum(policy_sample_weights), 1e-8)

    # Compute value loss with per-sample weighting
    # First compute per-sample BCE with masking
    bce = jnp.maximum(pred_value_logits, 0) - pred_value_logits * value_targets + jnp.log1p(jnp.exp(-jnp.abs(pred_value_logits)))
    masked_bce = bce * value_masks  # (batch, 4)
    value_loss_per_sample = masked_bce.sum(axis=-1) / jnp.maximum(value_masks.sum(axis=-1), 1.0)

    # Apply sample weights and average
    value_loss = jnp.sum(value_loss_per_sample * value_sample_weights) / jnp.maximum(jnp.sum(value_sample_weights), 1e-8)

    # Compute L2 regularization (not weighted by samples)
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda x: (x ** 2).sum(), params)
    )

    # Total loss
    loss = policy_loss + value_loss + l2_reg

    # Bearoff/non-bearoff split metrics without extra forward passes
    bearoff_mask_f = is_bearoff_mask.astype(jnp.float32)
    bearoff_count = jnp.sum(bearoff_mask_f)
    total_count = bearoff_mask_f.shape[0]
    non_bearoff_count = total_count - bearoff_count

    bearoff_value_loss = jnp.sum(value_loss_per_sample * bearoff_mask_f) / jnp.maximum(bearoff_count, 1.0)
    bearoff_policy_loss = jnp.sum(policy_loss_per_sample * bearoff_mask_f) / jnp.maximum(bearoff_count, 1.0)

    non_bearoff_mask_f = 1.0 - bearoff_mask_f
    non_bearoff_value_loss = jnp.sum(value_loss_per_sample * non_bearoff_mask_f) / jnp.maximum(non_bearoff_count, 1.0)
    non_bearoff_policy_loss = jnp.sum(policy_loss_per_sample * non_bearoff_mask_f) / jnp.maximum(non_bearoff_count, 1.0)

    # Compute metrics (unweighted for interpretability)
    pred_policy_probs = jax.nn.softmax(pred_policy_logits, axis=-1)
    pred_top1 = jnp.argmax(pred_policy_probs, axis=-1)
    target_top1 = jnp.argmax(experience.policy_weights, axis=-1)
    policy_accuracy = jnp.mean(pred_top1 == target_top1)

    # Value accuracy: did we predict win correctly?
    pred_value_probs = jax.nn.sigmoid(pred_value_logits)
    pred_win = pred_value_probs[:, 0] > 0.5
    target_win = value_targets[:, 0] > 0.5
    value_accuracy = jnp.mean(pred_win == target_win)

    # Per-output value losses for monitoring
    # Compute per-output BCE (unweighted by sample weights, but masked)
    mean_per_output_bce = (masked_bce.sum(axis=0)) / jnp.maximum(value_masks.sum(axis=0), 1.0)

    # Output names: win, gam_win_cond, gam_loss_cond, bg_rate
    value_loss_win = mean_per_output_bce[0]
    value_loss_gam_win_cond = mean_per_output_bce[1]
    value_loss_gam_loss_cond = mean_per_output_bce[2]
    value_loss_bg_rate = mean_per_output_bce[3]

    # Predicted output distribution (mean over batch)
    mean_pred_probs = pred_value_probs.mean(axis=0)  # (4,)

    # Target distribution (mean over batch)
    mean_target_probs = value_targets.mean(axis=0)  # (4,)

    # Compute equity from predictions for monitoring
    pred_equity = value_outputs_to_equity(pred_value_probs)
    mean_pred_equity = pred_equity.mean()

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        'policy_accuracy': policy_accuracy,
        'value_accuracy': value_accuracy,
        'bearoff_value_loss': bearoff_value_loss,
        'bearoff_policy_loss': bearoff_policy_loss,
        'non_bearoff_value_loss': non_bearoff_value_loss,
        'non_bearoff_policy_loss': non_bearoff_policy_loss,
        # Per-output value losses (4-way)
        'value_loss_win': value_loss_win,
        'value_loss_gam_win_cond': value_loss_gam_win_cond,
        'value_loss_gam_loss_cond': value_loss_gam_loss_cond,
        'value_loss_bg_rate': value_loss_bg_rate,
        # Predicted output probabilities (for calibration monitoring)
        'pred_prob_win': mean_pred_probs[0],
        'pred_prob_gam_win_cond': mean_pred_probs[1],
        'pred_prob_gam_loss_cond': mean_pred_probs[2],
        'pred_prob_bg_rate': mean_pred_probs[3],
        # Target probabilities (ground truth)
        'target_prob_win': mean_target_probs[0],
        'target_prob_gam_win_cond': mean_target_probs[1],
        'target_prob_gam_loss_cond': mean_target_probs[2],
        'target_prob_bg_rate': mean_target_probs[3],
        # Derived equity
        'pred_equity': mean_pred_equity,
    }
    return loss, (aux_metrics, updates)

from .base_worker import BaseWorker, WorkerStats
from ..serialization import (
    serialize_weights,
    deserialize_weights,
    serialize_warm_tree,
    experiences_to_numpy_batch,
)
from ..buffer.redis_buffer import RedisReplayBuffer
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint, WorkerPhase


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
        self.games_per_epoch = self.config.get('games_per_epoch', 10)
        # Training steps to run each epoch (triggered by games_per_epoch)
        self.steps_per_epoch = self.config.get('steps_per_epoch', 500)

        # Surprise-weighted sampling configuration
        # 0 = uniform sampling, 1 = fully surprise-weighted
        self.surprise_weight = self.config.get('surprise_weight', 0.5)

        # CPU pipeline tuning: how many threads to use for msgpack decode + stacking.
        # 0/1 disables parallel decode.
        self.decode_threads = int(self.config.get('decode_threads', 0))

        # Run multiple gradient steps on the same sampled batch to increase GPU
        # utilization when CPU-side deserialization is the bottleneck.
        self.batch_reuse_steps = int(self.config.get('batch_reuse_steps', 1))
        if self.batch_reuse_steps < 1:
            self.batch_reuse_steps = 1
        self._reuse_steps_remaining: int = 0
        self._reuse_prepared: Optional[Dict[str, Any]] = None

        # Prefetch batches in a background thread to overlap Redis I/O with GPU compute.
        # This is a best-effort throughput optimization; set to 0 to disable.
        self.prefetch_batches = int(self.config.get('prefetch_batches', 2))
        # Prefetch queue holds CPU numpy batches (already deserialized/stacked)
        # to overlap Redis I/O + deserialization with GPU compute.
        self._prefetch_queue: Optional["queue.Queue[Dict[str, np.ndarray]]"] = None
        self._prefetch_thread: Optional[threading.Thread] = None

        # Prometheus discovery registration needs periodic refresh while training
        # (long training batches can outlive the discovery TTL).
        self._metrics_port_bound: Optional[int] = None
        self._metrics_last_registration_time: float = 0.0

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
        self._bearoff_lookup: Optional[V4BearoffLookup] = None
        self._bearoff_table_path = self.config.get('bearoff_table_path')
        if self._bearoff_table_path is None:
            # Default path
            default_path = Path(__file__).parent.parent.parent / 'data' / 'bearoff_15.npy'
            if default_path.exists():
                self._bearoff_table_path = str(default_path)

        # NOTE: bearoff table is loaded lazily after acquiring the training lock,
        # to avoid multiple trainers OOM-ing the host by loading ~35GB each.

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

        # Single-trainer lock (cluster-wide): prevents multiple trainers from
        # simultaneously loading the bearoff table and/or fighting for the same GPU.
        self._training_lock_key = "bgai:training:lock"
        self._training_lock_ttl_seconds = int(self.config.get("training_lock_ttl_seconds", 90))
        self._training_lock_refresh_seconds = max(5, int(self._training_lock_ttl_seconds // 3))
        self._training_lock_held = False
        self._training_lock_thread: Optional[threading.Thread] = None

        # MLflow configuration
        self.mlflow_tracking_uri = self.config.get('mlflow_tracking_uri')
        self.mlflow_experiment_name = self.config.get('mlflow_experiment_name', 'bgai-training')
        self._mlflow_run = None

        # Neural network (lazy initialization)
        self._nn_model = None
        self._train_state = None
        self._env = None

        # Training state
        self._total_steps = 0
        self._rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

        # Collection-gated training stats
        self._training_stats = TrainingStats()

        # Training progress tracking for live Prometheus updates during long batches.
        self._last_progress_time = time.time()
        self._last_progress_steps = 0

    def _acquire_training_lock(self) -> None:
        """Ensure only one training worker runs at a time (cluster-wide)."""
        ttl = int(self._training_lock_ttl_seconds)
        if ttl <= 0:
            raise ValueError("training_lock_ttl_seconds must be > 0")

        ok = self.buffer.redis.set(
            self._training_lock_key,
            self.worker_id,
            nx=True,
            ex=ttl,
        )
        if not ok:
            holder = self.buffer.redis.get(self._training_lock_key)
            holder_str = holder.decode() if isinstance(holder, bytes) else str(holder)
            raise RuntimeError(
                f"Another training worker is already running (lock={self._training_lock_key}, holder={holder_str})"
            )

        self._training_lock_held = True

        def refresher():
            while self.running and self._training_lock_held:
                try:
                    current = self.buffer.redis.get(self._training_lock_key)
                    current_str = current.decode() if isinstance(current, bytes) else current
                    if current_str != self.worker_id:
                        # Someone else stole/overwrote the lock; stop training.
                        print(
                            f"Worker {self.worker_id}: Training lock lost to {current_str}, stopping..."
                        )
                        self.stop()
                        return
                    self.buffer.redis.expire(self._training_lock_key, ttl)
                except Exception:
                    pass
                time.sleep(self._training_lock_refresh_seconds)

        self._training_lock_thread = threading.Thread(
            target=refresher,
            name=f"{self.worker_id}-training-lock",
            daemon=True,
        )
        self._training_lock_thread.start()

    def _release_training_lock(self) -> None:
        if not self._training_lock_held:
            return

        self._training_lock_held = False
        try:
            # Delete only if we still own it.
            self.buffer.redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
                1,
                self._training_lock_key,
                self.worker_id,
            )
        except Exception:
            pass

    def _maybe_load_bearoff_table(self) -> None:
        if not (self._bearoff_table_path and self.bearoff_enabled):
            return
        if self._bearoff_table_np is not None and self._bearoff_lookup is not None:
            return

        bin_path = Path(self._bearoff_table_path + '.bin') if not str(self._bearoff_table_path).endswith('.bin') else Path(self._bearoff_table_path)
        required_bytes = bin_path.stat().st_size

        # Fail fast if the host doesn't have enough free RAM to safely load the bearoff table.
        # Loading uses `f.read()` which materializes the full byte buffer in memory.
        safety_bytes = int(2 * 1024**3)  # 2 GiB headroom for process/allocator overhead
        mem_available = self._get_mem_available_bytes()
        mem_total = self._get_mem_total_bytes()
        required_gb = required_bytes / 1e9
        avail_gb = mem_available / 1e9
        total_gb = mem_total / 1e9

        print(
            f"Worker {self.worker_id}: Bearoff table load requires ~{required_gb:.2f} GB "
            f"(+{safety_bytes/1e9:.2f} GB headroom); MemAvailable={avail_gb:.2f} GB, MemTotal={total_gb:.2f} GB"
        )
        if mem_available < required_bytes + safety_bytes:
            raise RuntimeError(
                f"Insufficient RAM to load bearoff table: need ~{(required_bytes + safety_bytes)/1e9:.2f} GB "
                f"available, have {avail_gb:.2f} GB (path={bin_path})"
            )

        try:
            self._bearoff_lookup, self._bearoff_table_np = self._load_bearoff_table(self._bearoff_table_path)
            self._bearoff_table = True
            shape_str = self._bearoff_table_np.shape
            size_gb = self._bearoff_table_np.nbytes / 1e9
            print(f"Worker {self.worker_id}: Loaded bearoff table from {self._bearoff_table_path}")
            print(f"Worker {self.worker_id}: Table shape: {shape_str}, size: {size_gb:.2f} GB")
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to load bearoff table: {e}")
            self._bearoff_table = None
            self._bearoff_table_np = None
            self._bearoff_lookup = None

    @staticmethod
    def _get_mem_available_bytes() -> int:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) < 2:
                        break
                    return int(parts[1]) * 1024
        raise RuntimeError("Could not read MemAvailable from /proc/meminfo")

    @staticmethod
    def _get_mem_total_bytes() -> int:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) < 2:
                        break
                    return int(parts[1]) * 1024
        raise RuntimeError("Could not read MemTotal from /proc/meminfo")

    def stop(self) -> None:
        """Stop the worker gracefully and release the training lock."""
        self._release_training_lock()
        super().stop()

    @property
    def worker_type(self) -> str:
        return 'training'

    # =========================================================================
    # Bearoff/Endgame Value Methods
    # =========================================================================

    @staticmethod
    def _reward_to_value_targets_np(reward: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Numpy implementation of `reward_to_value_targets` for batched rewards."""
        reward = reward.astype(np.float32, copy=False)
        did_win = reward > 0
        was_gammon = np.abs(reward) >= 2
        was_bg = np.abs(reward) >= 3

        win_target = np.where(did_win, 1.0, 0.0).astype(np.float32)
        gam_win_cond_target = np.where(was_gammon & did_win, 1.0, 0.0).astype(np.float32)
        gam_loss_cond_target = np.where(was_gammon & ~did_win, 1.0, 0.0).astype(np.float32)
        bg_rate_target = np.where(was_bg, 1.0, 0.0).astype(np.float32)
        targets = np.stack([win_target, gam_win_cond_target, gam_loss_cond_target, bg_rate_target], axis=-1)

        win_mask = np.ones_like(reward, dtype=np.float32)
        gam_win_mask = np.where(did_win, 1.0, 0.0).astype(np.float32)
        gam_loss_mask = np.where(~did_win, 1.0, 0.0).astype(np.float32)
        bg_mask = np.where(was_gammon, 1.0, 0.0).astype(np.float32)
        masks = np.stack([win_mask, gam_win_mask, gam_loss_mask, bg_mask], axis=-1)

        return targets, masks

    def _load_bearoff_table(self, path: str) -> Tuple[V4BearoffLookup, np.ndarray]:
        """Load bearoff table (V4 format only).

        This project is single-user; we fail fast instead of supporting legacy table formats.
        V4 uses a `{path}.bin` + `{path}.json` pair and stores conditional gammon
        probabilities matching the NN value head semantics.
        """
        # Check for V4 format (path without extension, with .bin and .json files)
        bin_path = Path(path + '.bin') if not path.endswith('.bin') else Path(path)
        json_path = Path(path + '.json') if not path.endswith('.json') else Path(path.replace('.bin', '.json'))

        if not (bin_path.exists() and json_path.exists()):
            raise FileNotFoundError(
                f"Bearoff V4 table not found: expected {bin_path} and {json_path}"
            )

        lookup = load_v4_bearoff(str(bin_path), str(json_path))
        return lookup, lookup.data

    def _get_bearoff_value_np(self, board: np.ndarray) -> float:
        """Get perfect bearoff equity from the table (numpy version).

        Equity is scaled to [-1, 1] using cubeless points (no cube).

        Args:
            board: Board array of shape (28,)
        Returns:
            Equity from current player's perspective.
        """
        if self._bearoff_table_np is None:
            return 0.0

        # Board is always in "current player = black" orientation (PGX flips board each turn).
        # Current player's home board is points 18-23 (with 23 closest to bear-off).
        # Opponent's home board is points 0-5 (with 0 closest to bear-off).
        cur_home = np.maximum(0, board[23:17:-1])   # point 1..6 for current player
        opp_home = np.maximum(0, -board[0:6])       # point 1..6 for opponent

        from bgai.endgame.indexing import position_to_index_fast

        x_idx = position_to_index_fast(cur_home)
        o_idx = position_to_index_fast(opp_home)

        if self._bearoff_lookup is None:
            return 0.0

        win, gam_win_cond, gam_loss_cond, _ = self._bearoff_lookup.get_4way_values(x_idx, o_idx)
        p_gammon_win = win * gam_win_cond
        p_gammon_loss = (1.0 - win) * gam_loss_cond
        p_win = win

        # Cubeless equity with gammons valued at 2 points
        equity = (2.0 * p_win - 1.0) + (p_gammon_win - p_gammon_loss)
        return float(equity)

    def _get_bearoff_target_probs(self, x_idx: int, o_idx: int) -> np.ndarray:
        """Get 4-way conditional probability targets for bearoff position.

        Returns:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
            - win: P(win)
            - gam_win_cond: P(gammon | win) - conditional probability
            - gam_loss_cond: P(gammon | loss) - conditional probability
            - bg_rate: P(backgammon | gammon) - always 0 for bearoff
        """
        if self._bearoff_lookup is None:
            return np.zeros(4, dtype=np.float32)
        return self._bearoff_lookup.get_4way_values(x_idx, o_idx)

    def _apply_bearoff_values_to_numpy_batch(
        self,
        observation_nn: np.ndarray,
        cur_player_id: np.ndarray,
        rewards: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Compute value targets/masks/weights and (optionally) override rewards for bearoff positions.

        This intentionally stays on CPU (numpy) to avoid device->host transfers which
        can dominate training time at large batch sizes.
        """
        batch_size = observation_nn.shape[0]
        cur_player_id = cur_player_id.astype(np.int32, copy=False)
        rewards = rewards.astype(np.float32, copy=False)

        target_reward = rewards[np.arange(batch_size), cur_player_id]
        value_targets_np, value_masks_np = self._reward_to_value_targets_np(target_reward)

        is_bearoff = np.zeros(batch_size, dtype=bool)
        bearoff_avg_gam_win_cond = np.array(0.0, dtype=np.float32)

        if self._bearoff_table_np is not None and self.bearoff_enabled:
            boards = np.round(observation_nn[:, :28] * 15).astype(np.int16, copy=False)

            bar_empty = (boards[:, 24] == 0) & (boards[:, 25] == 0)
            cur_outside = np.any(boards[:, 0:18] > 0, axis=1)
            opp_outside = np.any(boards[:, 6:24] < 0, axis=1)
            is_bearoff = bar_empty & ~cur_outside & ~opp_outside

            bearoff_idx = np.flatnonzero(is_bearoff)
            if bearoff_idx.size:
                from bgai.endgame.indexing import position_to_index_lut

                bearoff_boards = boards[bearoff_idx]
                cur_home = np.maximum(0, bearoff_boards[:, 23:17:-1]).astype(np.int32, copy=False)
                opp_home = np.maximum(0, -bearoff_boards[:, 0:6]).astype(np.int32, copy=False)

                x_idx = position_to_index_lut(cur_home)
                o_idx = position_to_index_lut(opp_home)

                if self._bearoff_lookup is None:
                    raise ValueError("Bearoff lookup not initialized")
                n = int(self._bearoff_lookup.n)

                i = np.minimum(x_idx, o_idx).astype(np.int64, copy=False)
                j = np.maximum(x_idx, o_idx).astype(np.int64, copy=False)
                flat_idx = i * n - (i * (i - 1)) // 2 + (j - i)

                raw = self._bearoff_lookup.data[flat_idx]
                perspective0 = (x_idx <= o_idx)

                gam_win_cond_u16 = np.where(perspective0, raw[:, 0], raw[:, 6]).astype(np.float32)
                gam_loss_cond_u16 = np.where(perspective0, raw[:, 1], raw[:, 7]).astype(np.float32)
                eq_cl_u16 = np.where(perspective0, raw[:, 2], raw[:, 8]).astype(np.float32)

                gam_win_cond = gam_win_cond_u16 / 65535.0
                gam_loss_cond = gam_loss_cond_u16 / 65535.0
                eq_cl = (eq_cl_u16 / 65535.0) * 2.0 - 1.0
                win = (eq_cl + 1.0) / 2.0

                bearoff_targets = np.stack(
                    [win, gam_win_cond, gam_loss_cond, np.zeros_like(win, dtype=np.float32)],
                    axis=-1,
                ).astype(np.float32)
                value_targets_np[bearoff_idx] = bearoff_targets

                has_win = win > 0.0
                has_loss = (1.0 - win) > 0.0
                has_gammon = (gam_win_cond > 0.0) | (gam_loss_cond > 0.0)
                value_masks_np[bearoff_idx] = np.stack(
                    [
                        np.ones_like(win, dtype=np.float32),
                        has_win.astype(np.float32),
                        has_loss.astype(np.float32),
                        has_gammon.astype(np.float32),
                    ],
                    axis=-1,
                )

                p_gammon_win = win * gam_win_cond
                p_gammon_loss = (1.0 - win) * gam_loss_cond
                equity = (2.0 * win - 1.0) + (p_gammon_win - p_gammon_loss)

                new_rewards = rewards.copy()
                cur_p = cur_player_id[bearoff_idx]
                opp_p = 1 - cur_p
                new_rewards[bearoff_idx, cur_p] = equity
                new_rewards[bearoff_idx, opp_p] = -equity
                rewards = new_rewards

                bearoff_avg_gam_win_cond = np.array(float(np.mean(gam_win_cond)), dtype=np.float32)

        value_sample_weights = np.ones(batch_size, dtype=np.float32)
        if is_bearoff.any():
            value_sample_weights[is_bearoff] = float(self.bearoff_value_weight)

        metrics = {
            'bearoff_count': np.array(int(np.sum(is_bearoff)), dtype=np.int32),
            'total_count': np.array(batch_size, dtype=np.int32),
            'is_bearoff_mask': is_bearoff,
            'bearoff_avg_gam_win_cond': bearoff_avg_gam_win_cond,
        }
        return rewards, metrics, value_targets_np, value_masks_np, value_sample_weights

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

                # Publish config to Redis so remote workers can fetch it
                self.state.set_config(self.config)
                print(f"Worker {self.worker_id}: Published config to Redis")

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
            """ResNet-style network with 4-way conditional value head for backgammon.

            Value head outputs logits for 4 independent probabilities (sigmoid):
            [win, gam_win_cond, gam_loss_cond, bg_rate]
            - win: P(current player wins)
            - gam_win_cond: P(gammon | win) - conditional probability
            - gam_loss_cond: P(gammon | loss) - conditional probability
            - bg_rate: P(backgammon | gammon) - combined rate
            """
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6
            value_head_out_size: int = 4  # 4-way conditional probabilities

            @nn.compact
            def __call__(self, x, train: bool = False):  # noqa: ARG002 - train required by interface
                del train  # unused but required by turbozero interface
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                # 4-way conditional value head: outputs logits, converted to probs via sigmoid
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
        self._last_progress_steps = self._total_steps
        self._last_progress_time = time.time()

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
            # Set warm tree building phase
            metrics = get_metrics()
            metrics.worker_phase.labels(
                worker_id=self.worker_id, worker_type='training'
            ).set(WorkerPhase.WARM_TREE)

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

            # Record warm tree phase duration
            metrics.phase_duration_seconds.labels(
                worker_id=self.worker_id, phase='warm_tree'
            ).observe(duration)

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
    def _train_step_weighted(
        self,
        train_state: TrainState,
        batch: BaseExperience,
        value_targets: jnp.ndarray,
        value_masks: jnp.ndarray,
        value_sample_weights: jnp.ndarray,
        policy_sample_weights: jnp.ndarray,
        is_bearoff_mask: jnp.ndarray,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Perform a single training step with per-sample weights.

        This enables weighting bearoff positions more heavily since they
        have known-perfect target values.

        Args:
            train_state: Current training state.
            batch: Batch of experiences.
            value_targets: Precomputed 4-way value targets.
            value_masks: Masks indicating which value targets are active.
            value_sample_weights: Per-sample weights for value loss (e.g., bearoff_value_weight).
            policy_sample_weights: Per-sample weights for policy loss.
            is_bearoff_mask: Boolean mask indicating bearoff positions.

        Returns:
            Tuple of (updated_train_state, metrics_dict).
        """
        loss_fn = partial(weighted_az_loss_fn, l2_reg_lambda=self.l2_reg_lambda)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, updates)), grads = grad_fn(
            train_state.params,
            train_state,
            batch,
            value_targets,
            value_masks,
            value_sample_weights,
            policy_sample_weights,
            is_bearoff_mask,
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
        if self.prefetch_batches > 0:
            self._ensure_prefetcher_started()

        # Check buffer size
        buffer_size = self.buffer.get_size()
        if buffer_size < self.min_buffer_size:
            return None

        prepared = None
        if self._reuse_steps_remaining > 0 and self._reuse_prepared is not None:
            prepared = self._reuse_prepared
            self._reuse_steps_remaining -= 1
        else:
            np_batch = self._get_next_np_batch()
            if not np_batch:
                return None

            required_keys = ('observation_nn', 'policy_weights', 'policy_mask', 'cur_player_id', 'reward')
            missing = [k for k in required_keys if k not in np_batch]
            if missing:
                print(f"Worker {self.worker_id}: Missing batch keys after conversion: {missing}")
                return None

            rewards_np, bearoff_metrics, value_targets_np, value_masks_np, value_sample_weights_np = (
                self._apply_bearoff_values_to_numpy_batch(
                    observation_nn=np_batch['observation_nn'],
                    cur_player_id=np_batch['cur_player_id'],
                    rewards=np_batch['reward'],
                )
            )

            jax_batch = BaseExperience(
                observation_nn=jnp.array(np_batch['observation_nn']),
                policy_weights=jnp.array(np_batch['policy_weights']),
                policy_mask=jnp.array(np_batch['policy_mask']),
                cur_player_id=jnp.array(np_batch['cur_player_id']),
                reward=jnp.array(rewards_np),
            )

            value_targets = jnp.array(value_targets_np, dtype=jnp.float32)
            value_masks = jnp.array(value_masks_np, dtype=jnp.float32)
            value_sample_weights = jnp.array(value_sample_weights_np, dtype=jnp.float32)
            is_bearoff_mask = jnp.array(bearoff_metrics['is_bearoff_mask'], dtype=jnp.bool_)

            prepared = {
                'jax_batch': jax_batch,
                'value_targets': value_targets,
                'value_masks': value_masks,
                'value_sample_weights': value_sample_weights,
                'is_bearoff_mask': is_bearoff_mask,
                'bearoff_metrics': bearoff_metrics,
            }

            if self.batch_reuse_steps > 1:
                self._reuse_prepared = prepared
                self._reuse_steps_remaining = self.batch_reuse_steps - 1
            else:
                self._reuse_prepared = None
                self._reuse_steps_remaining = 0

        jax_batch = prepared['jax_batch']
        value_targets = prepared['value_targets']
        value_masks = prepared['value_masks']
        value_sample_weights = prepared['value_sample_weights']
        is_bearoff_mask = prepared['is_bearoff_mask']
        bearoff_metrics = prepared['bearoff_metrics']

        # Keep policy samples unweighted; bearoff logits come from self-play
        policy_sample_weights = jnp.ones_like(value_sample_weights)

        # Perform training step - always use weighted version for 4-way value head
        self._train_state, metrics = self._train_step_weighted(
            self._train_state,
            jax_batch,
            value_targets,
            value_masks,
            value_sample_weights,
            policy_sample_weights,
            is_bearoff_mask,
        )

        # Convert metrics to Python floats
        metrics = {k: float(v) for k, v in metrics.items()}

        # Add bearoff metrics and compute separate value losses
        bearoff_count = int(bearoff_metrics['bearoff_count'])
        total_count = int(bearoff_metrics['total_count'])
        metrics['bearoff_count'] = bearoff_count
        metrics['bearoff_pct'] = 100.0 * bearoff_count / max(total_count, 1)
        metrics['bearoff_avg_gam_win_cond'] = float(bearoff_metrics['bearoff_avg_gam_win_cond'])

        # Split losses come from the jitted loss function (no extra forward passes).
        non_bearoff_count = total_count - bearoff_count
        if bearoff_count > 0:
            self._training_stats.cumulative_bearoff_value_loss += metrics.get('bearoff_value_loss', 0.0) * bearoff_count
            self._training_stats.bearoff_value_loss_count += bearoff_count
        if non_bearoff_count > 0:
            self._training_stats.cumulative_non_bearoff_value_loss += metrics.get('non_bearoff_value_loss', 0.0) * non_bearoff_count
            self._training_stats.non_bearoff_value_loss_count += non_bearoff_count

        # Update experience counts
        self._training_stats.bearoff_experiences += bearoff_count
        self._training_stats.non_bearoff_experiences += non_bearoff_count

        self._total_steps += 1
        self.stats.training_steps += 1

        return metrics

    def _ensure_prefetcher_started(self) -> None:
        if self._prefetch_thread is not None:
            return
        if self.prefetch_batches <= 0:
            return
        self._prefetch_queue = queue.Queue(maxsize=max(1, self.prefetch_batches))

        def _runner():
            while self.running:
                try:
                    # Check buffer size before trying to sample.
                    if self.buffer.get_size() < self.min_buffer_size:
                        time.sleep(0.2)
                        continue

                    # Sample batch from Redis with surprise-weighted sampling.
                    min_version = max(0, self.current_model_version - 10)
                    if self.surprise_weight > 0:
                        batch_data = self.buffer.sample_batch_surprise_weighted(
                            self.train_batch_size,
                            surprise_weight=self.surprise_weight,
                            min_model_version=min_version,
                        )
                    else:
                        batch_data = self.buffer.sample_batch(
                            self.train_batch_size,
                            min_model_version=min_version,
                            require_rewards=True,
                        )

                    if len(batch_data) < self.train_batch_size:
                        time.sleep(0.05)
                        continue

                    np_batch = experiences_to_numpy_batch(
                        batch_data,
                        decode_threads=self.decode_threads,
                    )
                    if not np_batch:
                        time.sleep(0.05)
                        continue

                    required_keys = (
                        'observation_nn',
                        'policy_weights',
                        'policy_mask',
                        'cur_player_id',
                        'reward',
                    )
                    if any(k not in np_batch for k in required_keys):
                        time.sleep(0.05)
                        continue

                    # Best-effort: drop if queue is full.
                    assert self._prefetch_queue is not None
                    try:
                        self._prefetch_queue.put(np_batch, timeout=0.1)
                    except queue.Full:
                        pass
                except Exception:
                    time.sleep(0.1)

        self._prefetch_thread = threading.Thread(
            target=_runner,
            name=f"training-prefetch-{self.worker_id}",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _get_next_np_batch(self) -> Optional[Dict[str, np.ndarray]]:
        if self._prefetch_queue is not None:
            try:
                return self._prefetch_queue.get_nowait()
            except queue.Empty:
                pass

        # Fallback to synchronous sampling.
        min_version = max(0, self.current_model_version - 10)  # Allow slightly old experiences
        if self.surprise_weight > 0:
            batch_data = self.buffer.sample_batch_surprise_weighted(
                self.train_batch_size,
                surprise_weight=self.surprise_weight,
                min_model_version=min_version,
            )
        else:
            batch_data = self.buffer.sample_batch(
                self.train_batch_size,
                min_model_version=min_version,
                require_rewards=True,
            )

        if len(batch_data) < self.train_batch_size:
            return None

        np_batch = experiences_to_numpy_batch(
            batch_data,
            decode_threads=self.decode_threads,
        )
        return np_batch or None

    def _maybe_refresh_metrics_registration(
        self,
        ttl_seconds: int = 300,
        min_interval_seconds: float = 60.0,
    ) -> None:
        if self._metrics_port_bound is None:
            return
        now = time.time()
        if (now - self._metrics_last_registration_time) < min_interval_seconds:
            return
        try:
            register_metrics_endpoint(
                self.buffer.redis,
                worker_id=self.worker_id,
                worker_type='training',
                port=int(self._metrics_port_bound),
                ttl_seconds=int(ttl_seconds),
            )
            self._metrics_last_registration_time = now
        except Exception:
            pass

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
        last_stdout_time = batch_start

        self._training_stats.current_batch_start_time = batch_start
        self._training_stats.current_batch_steps = 0

        prom_metrics = get_metrics()

        while steps_done < target_steps and self.running:
            # Keep Prometheus discovery alive during long training batches.
            self._maybe_refresh_metrics_registration()

            step_metrics = self._sample_and_train()

            if step_metrics is None:
                # Not enough valid experiences, wait briefly
                time.sleep(0.1)
                continue

            batch_metrics.append(step_metrics)
            steps_done += 1
            self._training_stats.current_batch_steps = steps_done
            self._training_stats.cumulative_loss += step_metrics.get('loss', 0)
            self._training_stats.loss_count += 1

            # Persist step counter to Redis periodically for recovery
            if self._total_steps % 100 == 0:
                self.state.set_training_steps(self._total_steps)
                now = time.time()
                dt = now - self._last_progress_time
                if dt > 0:
                    live_steps_per_sec = (self._total_steps - self._last_progress_steps) / dt
                    prom_metrics.training_steps_per_second.labels(
                        worker_id=self.worker_id
                    ).set(live_steps_per_sec)
                    self._last_progress_time = now
                    self._last_progress_steps = self._total_steps

            now = time.time()
            if now - last_stdout_time >= 30.0:
                elapsed = now - batch_start
                steps_per_sec = steps_done / max(elapsed, 0.001)
                avg_loss = (
                    self._training_stats.cumulative_loss / max(self._training_stats.loss_count, 1)
                )
                print(
                    f"Worker {self.worker_id}: Training progress "
                    f"{steps_done}/{target_steps} steps "
                    f"(total_steps={self._total_steps}, {steps_per_sec:.2f} steps/s, avg_loss={avg_loss:.4f})"
                )
                last_stdout_time = now

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
            expected_keys = set(batch_metrics[0].keys())
            for i, metric_dict in enumerate(batch_metrics[1:], start=1):
                keys = set(metric_dict.keys())
                if keys != expected_keys:
                    missing = sorted(expected_keys - keys)
                    extra = sorted(keys - expected_keys)
                    raise RuntimeError(
                        f"Training step metrics schema mismatch at step {i}: "
                        f"missing={missing}, extra={extra}"
                    )

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

        # Ensure only one training worker is active cluster-wide (prevents RAM OOM).
        self._acquire_training_lock()

        # Now that we own the training lock, load large resources (bearoff table).
        self._maybe_load_bearoff_table()

        # Ensure collection is not paused from a previous crash
        self.state.set_collection_paused(False)
        print(f"Worker {self.worker_id}: Cleared collection pause state")

        # Set up MLflow tracking
        self._setup_mlflow()

        # Start Prometheus metrics server
        metrics_port_config = self.config.get('metrics_port', 9200)
        metrics_port = start_metrics_server(metrics_port_config)
        if metrics_port is None:
            print(f"Worker {self.worker_id}: Failed to start metrics server")
            metrics_port = metrics_port_config  # Fallback for registration
        metrics = get_metrics()

        self._metrics_port_bound = int(metrics_port)
        self._metrics_last_registration_time = 0.0
        last_metrics_refresh = time.time()

        # Register metrics endpoint for dynamic discovery (use actual bound port)
        try:
            register_metrics_endpoint(
                self.buffer.redis,
                worker_id=self.worker_id,
                worker_type='training',
                port=metrics_port,
                ttl_seconds=300,
            )
            self._metrics_last_registration_time = time.time()
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

        steps_per_game_target = self.config.get(
            'steps_per_game',
            self.steps_per_epoch / max(self.games_per_epoch, 1),
        )
        min_backlog_steps = int(self.config.get('min_backlog_steps', 1))
        max_train_steps_per_batch = int(self.config.get('max_train_steps_per_batch', self.steps_per_epoch))
        # Only pause collection when training uses a GPU; CPU training does not contend
        # with GPU collection workers.
        pause_collection_during_training = (
            bool(self.config.get('pause_collection_during_training', False))
            and self.device_info.is_gpu
        )

        print(f"Worker {self.worker_id}: Game-ratio-gated training mode")
        print(
            f"Worker {self.worker_id}: Target {steps_per_game_target:.3f} train steps/game "
            f"(steps_per_epoch={self.steps_per_epoch}, games_per_epoch={self.games_per_epoch})"
        )
        print(
            f"Worker {self.worker_id}: Will train when backlog >= {min_backlog_steps} "
            f"(max per batch: {max_train_steps_per_batch})"
        )
        if pause_collection_during_training:
            print(f"Worker {self.worker_id}: Will pause collection during training (exclusive GPU)")
        else:
            print(f"Worker {self.worker_id}: Will NOT pause collection during training (shared GPU)")

        start_time = time.time()
        last_log_time = start_time
        last_mlflow_heartbeat_time = start_time
        phase_start_time = start_time
        current_phase = WorkerPhase.IDLE

        def set_phase(phase: int, phase_name: str = None):
            """Set worker phase and record duration of previous phase."""
            nonlocal phase_start_time, current_phase
            # Record duration of previous phase
            phase_duration = time.time() - phase_start_time
            if current_phase != WorkerPhase.IDLE:  # Don't record idle durations
                phase_names = {
                    WorkerPhase.TRAINING: 'training',
                    WorkerPhase.WARM_TREE: 'warm_tree',
                    WorkerPhase.CHECKPOINT: 'checkpoint',
                    WorkerPhase.SAMPLING: 'sampling',
                }
                prev_phase_name = phase_names.get(current_phase, 'unknown')
                metrics.phase_duration_seconds.labels(
                    worker_id=self.worker_id, phase=prev_phase_name
                ).observe(phase_duration)
            # Set new phase
            current_phase = phase
            phase_start_time = time.time()
            metrics.worker_phase.labels(
                worker_id=self.worker_id, worker_type='training'
            ).set(phase)

        # Initialize tracking
        self._training_stats.games_at_last_train = self._get_current_games_count()
        self._training_stats.experiences_at_last_train = self.buffer.get_size()
        set_phase(WorkerPhase.IDLE)

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

            # Determine training target based on collected games, and train to catch up.
            current_games = self._get_current_games_count()
            target_total_steps = int(math.floor(current_games * steps_per_game_target))
            backlog_steps = target_total_steps - self._total_steps
            new_games = current_games - self._training_stats.games_at_last_train

            metrics.training_target_steps_total.labels(worker_id=self.worker_id).set(target_total_steps)
            metrics.training_backlog_steps.labels(worker_id=self.worker_id).set(backlog_steps)
            metrics.games_since_last_train.labels(worker_id=self.worker_id).set(max(new_games, 0))

            if backlog_steps < min_backlog_steps:
                # Caught up (or very small backlog), wait briefly.
                time.sleep(1.0)

                # Keep training visible in Prometheus discovery even while idle.
                current_time = time.time()
                if current_time - last_metrics_refresh >= 60.0:
                    try:
                        register_metrics_endpoint(
                            self.buffer.redis,
                            worker_id=self.worker_id,
                            worker_type='training',
                            port=metrics_port,
                            ttl_seconds=300,
                        )
                        self._metrics_last_registration_time = current_time
                    except Exception:
                        pass
                    metrics.training_steps_per_second.labels(
                        worker_id=self.worker_id
                    ).set(0.0)
                    last_metrics_refresh = current_time

                # Periodic status log while waiting
                if current_time - last_log_time >= 30.0:
                    print(
                        f"Worker {self.worker_id}: Waiting (backlog={backlog_steps}, "
                        f"target_steps={target_total_steps}, "
                        f"new_games={new_games}), "
                        f"total_games={current_games}, "
                        f"buffer={buffer_size}, "
                        f"version={self.current_model_version}"
                    )
                    last_log_time = current_time

                # MLflow heartbeat: enables plotting overall progress vs wall time.
                if current_time - last_mlflow_heartbeat_time >= 30.0:
                    self._log_mlflow_metrics(
                        {
                            'progress_total_train_steps': float(self._total_steps),
                            'progress_total_games': float(current_games),
                            'progress_buffer_size': float(buffer_size),
                            'progress_new_games_since_train': float(new_games),
                            'progress_target_train_steps': float(target_total_steps),
                            'progress_training_backlog_steps': float(backlog_steps),
                            'progress_train_steps_per_sec': 0.0,
                        },
                        step=int(current_time),
                    )
                    last_mlflow_heartbeat_time = current_time
                continue

            # Train a bounded chunk of the backlog.
            target_steps = min(max_train_steps_per_batch, max(backlog_steps, 0))

            print(
                f"Worker {self.worker_id}: Training batch triggered! "
                f"backlog={backlog_steps} -> {target_steps} training steps "
                f"(target_total_steps={target_total_steps}, current_steps={self._total_steps}, "
                f"new_games={new_games})"
            )

            paused_collection = False
            try:
                if pause_collection_during_training:
                    # Pause game collection for exclusive GPU access during training
                    self.state.set_collection_paused(True)
                    paused_collection = True
                    print(f"Worker {self.worker_id}: Paused game collection for training")

                # Set training phase
                set_phase(WorkerPhase.TRAINING)

                # Run training epoch (game collection is paused)
                batch_metrics = self._run_training_batch(target_steps)

                # Push updated weights to Redis
                self._push_weights_to_redis()

            finally:
                # Back to idle even if training fails (prevents deadlocked collection pause).
                set_phase(WorkerPhase.IDLE)

                if paused_collection:
                    self.state.set_collection_paused(False)
                    print(f"Worker {self.worker_id}: Resumed game collection")

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
                # 4. Per-output value metrics (4-way conditional format)
                'pred_win': batch_metrics.get('pred_win', 0),
                'pred_gam_win_cond': batch_metrics.get('pred_gam_win_cond', 0),
                'pred_gam_loss_cond': batch_metrics.get('pred_gam_loss_cond', 0),
                'pred_bg_rate': batch_metrics.get('pred_bg_rate', 0),
                'target_win': batch_metrics.get('target_win', 0),
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

            # Per-output value metrics (4-way conditional format)
            output_names = ['win', 'gam_win_cond', 'gam_loss_cond', 'bg_rate']
            for output_name in output_names:
                metrics.predicted_outcome_prob.labels(
                    worker_id=self.worker_id, outcome=output_name
                ).set(batch_metrics.get(f'pred_{output_name}', 0))

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
            'games_per_epoch': self.games_per_epoch,
            'steps_per_epoch': self.steps_per_epoch,
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
