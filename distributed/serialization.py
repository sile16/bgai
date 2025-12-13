"""Serialization utilities for distributed training.

This module provides functions for serializing and deserializing:
- Model weights (JAX pytrees) for weight synchronization
- Game experiences for Redis storage
- Checkpoint metadata

Uses msgpack for fast serialization of experiences and pickle for complex
nested structures like model parameters.
"""

import io
import pickle
from typing import Dict, Any, List, Optional

import numpy as np

# Lazy imports to avoid issues if packages not installed
_msgpack = None
_jax = None
_jnp = None
_decode_executor = None
_decode_executor_workers = 0
_decode_executor_lock = None


def _get_msgpack():
    """Lazy import msgpack with numpy support."""
    global _msgpack
    if _msgpack is None:
        import msgpack
        import msgpack_numpy as m
        m.patch()  # Enable numpy array support
        _msgpack = msgpack
    return _msgpack


def _get_jax():
    """Lazy import JAX."""
    global _jax, _jnp
    if _jax is None:
        import jax
        import jax.numpy as jnp
        _jax = jax
        _jnp = jnp
    return _jax, _jnp


def _get_decode_executor(max_workers: int):
    """Return a shared ThreadPoolExecutor for decode-heavy CPU work."""
    global _decode_executor, _decode_executor_workers, _decode_executor_lock
    if max_workers <= 1:
        return None

    if _decode_executor_lock is None:
        import threading
        _decode_executor_lock = threading.Lock()

    with _decode_executor_lock:
        if _decode_executor is not None and _decode_executor_workers == max_workers:
            return _decode_executor

        # Replace the existing executor if the worker count changes.
        # We don't aggressively shutdown the old pool to keep this safe for long-running processes.
        from concurrent.futures import ThreadPoolExecutor
        _decode_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bgai-decode")
        _decode_executor_workers = max_workers
        return _decode_executor


# =============================================================================
# Model Weight Serialization
# =============================================================================

def serialize_weights(params: Dict[str, Any]) -> bytes:
    """Serialize model weights for transmission between workers.

    Converts JAX arrays to numpy arrays and uses pickle for the nested
    parameter dictionary structure.

    Args:
        params: Model parameters as a nested dict (e.g., {'params': {...}})

    Returns:
        Serialized bytes that can be transmitted or stored.

    Example:
        >>> weights_bytes = serialize_weights({'params': model.params})
        >>> # Send weights_bytes to another worker
    """
    jax, jnp = _get_jax()

    def to_numpy(x):
        """Convert JAX arrays to numpy arrays."""
        if hasattr(x, 'device'):  # JAX array
            return np.array(x)
        return x

    numpy_params = jax.tree_util.tree_map(to_numpy, params)

    buffer = io.BytesIO()
    pickle.dump(numpy_params, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.getvalue()


def deserialize_weights(data: bytes) -> Dict[str, Any]:
    """Deserialize model weights received from another worker.

    Converts numpy arrays back to JAX arrays on the current device.

    Args:
        data: Serialized weight bytes from serialize_weights()

    Returns:
        Model parameters as a nested dict with JAX arrays.

    Example:
        >>> params = deserialize_weights(weights_bytes)
        >>> # Use params in model.apply()
    """
    jax, jnp = _get_jax()

    buffer = io.BytesIO(data)
    numpy_params = pickle.load(buffer)

    def to_jax(x):
        """Convert numpy arrays to JAX arrays."""
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x

    return jax.tree_util.tree_map(to_jax, numpy_params)


# =============================================================================
# Experience Serialization (for Redis)
# =============================================================================

def serialize_experience(experience: Dict[str, Any]) -> bytes:
    """Serialize a single game experience for Redis storage.

    Uses msgpack for fast, compact serialization. JAX arrays are
    converted to numpy arrays automatically.

    Args:
        experience: Dict containing game experience data:
            - observation_nn: Board observation (JAX array)
            - policy_weights: MCTS policy distribution (JAX array)
            - policy_mask: Legal action mask (JAX array)
            - cur_player_id: Current player (int or JAX scalar)
            - reward: Optional final rewards (JAX array or None)

    Returns:
        Msgpack-serialized bytes.

    Example:
        >>> exp = {'observation_nn': obs, 'policy_weights': policy, ...}
        >>> exp_bytes = serialize_experience(exp)
        >>> redis.set('exp:123', exp_bytes)
    """
    jax, jnp = _get_jax()
    msgpack = _get_msgpack()

    serializable = {}
    for key, value in experience.items():
        if value is None:
            serializable[key] = None
        elif hasattr(value, 'device'):  # JAX array
            serializable[key] = np.array(value)
        elif isinstance(value, (np.ndarray, np.generic)):
            serializable[key] = value
        else:
            # Scalars, strings, etc.
            serializable[key] = value

    return msgpack.packb(serializable, use_bin_type=True)


def deserialize_experience(data: bytes) -> Dict[str, Any]:
    """Deserialize a single experience from Redis.

    Returns numpy arrays (not JAX arrays) for efficiency when
    batching multiple experiences.

    Args:
        data: Msgpack-serialized bytes from serialize_experience()

    Returns:
        Dict with numpy arrays.
    """
    msgpack = _get_msgpack()
    return msgpack.unpackb(data, raw=False)


def serialize_rewards(rewards: Any) -> bytes:
    """Serialize final game rewards.

    Args:
        rewards: JAX array of shape (num_players,) with final rewards.

    Returns:
        Msgpack-serialized bytes.
    """
    jax, jnp = _get_jax()
    msgpack = _get_msgpack()

    if hasattr(rewards, 'device'):
        rewards = np.array(rewards)

    return msgpack.packb(rewards, use_bin_type=True)


def deserialize_rewards(data: bytes) -> np.ndarray:
    """Deserialize final game rewards.

    Args:
        data: Msgpack-serialized bytes.

    Returns:
        Numpy array of rewards.
    """
    msgpack = _get_msgpack()
    return msgpack.unpackb(data, raw=False)


# =============================================================================
# Batch Conversion
# =============================================================================

def batch_experiences_to_jax(experiences: List[Dict[str, Any]]):
    """Convert a list of experiences to a batched BaseExperience dataclass.

    Takes experiences retrieved from Redis and stacks them into
    batched arrays suitable for training. Returns a BaseExperience
    dataclass compatible with az_default_loss_fn.

    Args:
        experiences: List of experience dicts, each containing:
            - 'data': Serialized experience bytes
            - 'final_rewards': Optional serialized rewards bytes
            - 'model_version': int

    Returns:
        BaseExperience dataclass with batched JAX arrays:
            - observation_nn: (batch_size, obs_dim)
            - policy_weights: (batch_size, num_actions)
            - policy_mask: (batch_size, num_actions)
            - cur_player_id: (batch_size,)
            - reward: (batch_size, num_players)

    Example:
        >>> raw_batch = redis_buffer.sample_batch(128)
        >>> batch = batch_experiences_to_jax(raw_batch)
        >>> loss = train_step(params, batch)
    """
    from core.memory.replay_memory import BaseExperience

    jax, jnp = _get_jax()

    # Deserialize all experiences
    deserialized = []
    for exp in experiences:
        if isinstance(exp.get('data'), bytes):
            exp_data = deserialize_experience(exp['data'])
        else:
            exp_data = exp.get('data', exp)

        # Add final rewards if available
        final_rewards = exp.get('final_rewards')
        if final_rewards is not None:
            if isinstance(final_rewards, bytes):
                rewards = deserialize_rewards(final_rewards)
            elif isinstance(final_rewards, (np.ndarray, np.generic, int, float)) or hasattr(final_rewards, '__len__'):
                rewards = final_rewards
            else:
                raise TypeError(f"Unsupported final_rewards type: {type(final_rewards)}")

            rewards = np.asarray(rewards)
            if rewards.shape == ():
                # Scalar reward is assumed to be player-0 perspective.
                r0 = float(rewards)
                rewards = np.array([r0, -r0], dtype=np.float32)
            elif rewards.ndim == 1 and rewards.shape[0] == 2:
                rewards = rewards.astype(np.float32, copy=False)
                # We expect zero-sum terminal rewards so cur_player selection is correct.
                if not np.isfinite(rewards).all():
                    raise ValueError(f"Non-finite terminal rewards: {rewards!r}")
                if abs(float(rewards[0] + rewards[1])) > 1e-4:
                    raise ValueError(
                        f"Expected zero-sum terminal rewards [r0, r1] with r0==-r1, got {rewards!r}"
                    )
            else:
                raise ValueError(f"Unexpected terminal rewards shape: {rewards.shape!r}")

            exp_data['reward'] = rewards

        deserialized.append(exp_data)

    if not deserialized:
        return {}

    # Stack into batched arrays
    batch = {}
    sample_keys = deserialized[0].keys()

    for key in sample_keys:
        values = [exp.get(key) for exp in deserialized]
        # Filter out None values
        values = [v for v in values if v is not None]

        if values:
            # Stack numpy arrays
            if isinstance(values[0], np.ndarray):
                stacked = np.stack(values, axis=0)
                batch[key] = jnp.array(stacked)
            elif isinstance(values[0], (int, float)):
                batch[key] = jnp.array(values)
            else:
                # Keep as-is for other types
                batch[key] = values

    # Convert dict to BaseExperience dataclass
    return BaseExperience(
        observation_nn=batch.get('observation_nn'),
        policy_weights=batch.get('policy_weights'),
        policy_mask=batch.get('policy_mask'),
        cur_player_id=batch.get('cur_player_id'),
        reward=batch.get('reward'),
    )


def experiences_to_numpy_batch(
    experiences: List[Dict[str, Any]],
    decode_threads: int = 0,
) -> Dict[str, np.ndarray]:
    """Convert experiences to batched numpy arrays (no JAX conversion).

    Useful when you want to stay on CPU or defer JAX conversion.

    Args:
        experiences: List of experience dicts with serialized data.
        decode_threads: If >1, decode experiences in a shared thread pool.

    Returns:
        Dict with batched numpy arrays.
    """
    # Ensure msgpack_numpy patch is applied before any parallel decode.
    _get_msgpack()

    def _decode_one(exp: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(exp.get('data'), bytes):
            exp_data = deserialize_experience(exp['data'])
        else:
            exp_data = exp.get('data', exp)

        if exp.get('final_rewards'):
            if isinstance(exp['final_rewards'], bytes):
                rewards = deserialize_rewards(exp['final_rewards'])
            else:
                rewards = exp['final_rewards']

            rewards = np.asarray(rewards)
            if rewards.shape == ():
                r0 = float(rewards)
                rewards = np.array([r0, -r0], dtype=np.float32)
            elif rewards.ndim == 1 and rewards.shape[0] == 2:
                rewards = rewards.astype(np.float32, copy=False)
                if not np.isfinite(rewards).all():
                    raise ValueError(f"Non-finite terminal rewards: {rewards!r}")
                if abs(float(rewards[0] + rewards[1])) > 1e-4:
                    raise ValueError(
                        f"Expected zero-sum terminal rewards [r0, r1] with r0==-r1, got {rewards!r}"
                    )
            else:
                raise ValueError(f"Unexpected terminal rewards shape: {rewards.shape!r}")

            exp_data['reward'] = rewards

        return exp_data

    # Deserialize all experiences
    if decode_threads and decode_threads > 1 and len(experiences) >= 256:
        executor = _get_decode_executor(int(decode_threads))
        if executor is None:
            deserialized = [_decode_one(exp) for exp in experiences]
        else:
            deserialized = list(executor.map(_decode_one, experiences))
    else:
        deserialized = [_decode_one(exp) for exp in experiences]

    if not deserialized:
        return {}

    # Stack into batched arrays
    batch = {}
    sample_keys = deserialized[0].keys()

    for key in sample_keys:
        values = [exp.get(key) for exp in deserialized]
        values = [v for v in values if v is not None]

        if values:
            if isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values, axis=0)
            elif isinstance(values[0], (int, float)):
                batch[key] = np.array(values)
            else:
                batch[key] = values

    return batch


# =============================================================================
# Checkpoint Metadata
# =============================================================================

def serialize_checkpoint_metadata(metadata: Dict[str, Any]) -> bytes:
    """Serialize checkpoint metadata.

    Args:
        metadata: Dict with checkpoint info (step, timestamp, etc.)

    Returns:
        Msgpack-serialized bytes.
    """
    msgpack = _get_msgpack()
    return msgpack.packb(metadata, use_bin_type=True)


def deserialize_checkpoint_metadata(data: bytes) -> Dict[str, Any]:
    """Deserialize checkpoint metadata.

    Args:
        data: Msgpack-serialized bytes.

    Returns:
        Metadata dict.
    """
    msgpack = _get_msgpack()
    return msgpack.unpackb(data, raw=False)


# =============================================================================
# Warm Tree Serialization (for MCTS tree sharing)
# =============================================================================

def serialize_warm_tree(tree_state: Any) -> bytes:
    """Serialize an MCTS tree state for sharing across workers.

    The warm tree is a pre-computed deep MCTS tree from the initial game
    position that workers can use as a starting point.

    Args:
        tree_state: MCTSTree state from the evaluator.

    Returns:
        Serialized bytes that can be stored in Redis.

    Example:
        >>> warm_tree = mcts_evaluator.init(template_state)
        >>> # Run deep MCTS
        >>> for _ in range(5000):
        ...     warm_tree = mcts_evaluator.iterate(key, warm_tree, params, step_fn)
        >>> tree_bytes = serialize_warm_tree(warm_tree)
    """
    jax, jnp = _get_jax()

    def to_numpy(x):
        """Convert JAX arrays to numpy arrays."""
        if hasattr(x, 'device'):  # JAX array
            return np.array(x)
        return x

    numpy_tree = jax.tree_util.tree_map(to_numpy, tree_state)

    buffer = io.BytesIO()
    pickle.dump(numpy_tree, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.getvalue()


def deserialize_warm_tree(data: bytes) -> Any:
    """Deserialize a warm MCTS tree state.

    Converts numpy arrays back to JAX arrays on the current device.

    Args:
        data: Serialized tree bytes from serialize_warm_tree()

    Returns:
        MCTSTree state ready for use with the evaluator.

    Example:
        >>> tree_bytes = redis.get('bgai:model:warm_tree')
        >>> warm_tree = deserialize_warm_tree(tree_bytes)
        >>> # Use warm_tree as initial eval_state
    """
    jax, jnp = _get_jax()

    buffer = io.BytesIO(data)
    numpy_tree = pickle.load(buffer)

    def to_jax(x):
        """Convert numpy arrays to JAX arrays."""
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x

    return jax.tree_util.tree_map(to_jax, numpy_tree)


# =============================================================================
# Utility Functions
# =============================================================================

def get_serialized_size(data: bytes) -> str:
    """Get human-readable size of serialized data.

    Args:
        data: Serialized bytes.

    Returns:
        Human-readable size string (e.g., "1.23 MB").
    """
    size = len(data)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def estimate_experience_size(
    obs_dim: int = 34,
    num_actions: int = 156,
    num_players: int = 2,
    dtype_bytes: int = 4  # float32
) -> int:
    """Estimate the serialized size of a single experience.

    Args:
        obs_dim: Observation dimension.
        num_actions: Number of actions.
        num_players: Number of players.
        dtype_bytes: Bytes per element (4 for float32).

    Returns:
        Estimated size in bytes.
    """
    # observation_nn
    size = obs_dim * dtype_bytes
    # policy_weights
    size += num_actions * dtype_bytes
    # policy_mask (bool = 1 byte)
    size += num_actions
    # cur_player_id (int32 = 4 bytes)
    size += 4
    # reward
    size += num_players * dtype_bytes
    # msgpack overhead (~10-20%)
    size = int(size * 1.15)

    return size
