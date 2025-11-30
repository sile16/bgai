"""Redis-backed replay buffer for distributed training.

This module provides a replay buffer implementation that stores experiences
in Redis, allowing multiple workers to contribute and sample from a shared
buffer across the network.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..serialization import (
    serialize_experience,
    deserialize_experience,
    serialize_rewards,
    deserialize_rewards,
)


@dataclass
class BufferStats:
    """Statistics about the replay buffer."""
    total_experiences: int
    total_episodes: int
    capacity: int
    fullness_pct: float
    oldest_timestamp: Optional[float]
    newest_timestamp: Optional[float]
    experiences_with_rewards_pct: float


class RedisReplayBuffer:
    """Redis-backed replay buffer for distributed experience storage.

    Experiences are stored as Redis hashes with the following schema:
    - experience:{id} -> Hash with data, model_version, timestamp, etc.
    - episode:{id} -> Hash with episode metadata
    - buffer:experiences -> List of experience IDs (for FIFO eviction)
    - buffer:metadata -> Hash with buffer statistics

    Example:
        >>> buffer = RedisReplayBuffer(host='localhost', port=6379, worker_id='game-001')
        >>> buffer.add_experience(serialize_experience(exp), model_version=1)
        >>> batch = buffer.sample_batch(128)
    """

    # Redis key prefixes
    EXPERIENCE_KEY = "bgai:experience:"
    EPISODE_KEY = "bgai:episode:"
    EXPERIENCE_LIST = "bgai:buffer:experiences"
    EPISODE_LIST = "bgai:buffer:episodes"
    METADATA_KEY = "bgai:buffer:metadata"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        worker_id: Optional[str] = None,
        capacity: int = 100000,
        episode_capacity: int = 5000,
    ):
        """Initialize Redis replay buffer.

        Args:
            host: Redis server hostname.
            port: Redis server port.
            db: Redis database number.
            password: Optional Redis password.
            worker_id: Unique identifier for this worker.
            capacity: Maximum number of experiences to store.
            episode_capacity: Maximum number of episodes to store.
        """
        import redis

        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # Keep binary data
        )
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.capacity = capacity
        self.episode_capacity = episode_capacity

        # Verify connection
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Redis connection is working."""
        try:
            self.redis.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    def _generate_id(self) -> str:
        """Generate a unique ID for an experience or episode."""
        timestamp_us = int(time.time() * 1_000_000)
        return f"{self.worker_id}:{timestamp_us}"

    # =========================================================================
    # Adding Experiences
    # =========================================================================

    def add_experience(
        self,
        experience_bytes: bytes,
        model_version: int,
        episode_id: Optional[str] = None,
        has_reward: bool = False,
        final_rewards: Optional[bytes] = None,
    ) -> str:
        """Add a single experience to the buffer.

        Args:
            experience_bytes: Serialized experience from serialize_experience().
            model_version: Model version used to generate this experience.
            episode_id: Optional episode ID to link experiences.
            has_reward: Whether this experience has final rewards.
            final_rewards: Serialized final rewards if has_reward is True.

        Returns:
            Experience ID.

        Example:
            >>> exp_bytes = serialize_experience(exp)
            >>> exp_id = buffer.add_experience(exp_bytes, model_version=5)
        """
        exp_id = self._generate_id()
        exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

        # Store experience data
        mapping = {
            b'data': experience_bytes,
            b'model_version': str(model_version).encode(),
            b'timestamp': str(time.time()).encode(),
            b'has_reward': b'1' if has_reward else b'0',
            b'episode_id': (episode_id or '').encode(),
        }

        if final_rewards:
            mapping[b'final_rewards'] = final_rewards

        self.redis.hset(exp_key, mapping=mapping)

        # Add to experience list for FIFO tracking
        self.redis.lpush(self.EXPERIENCE_LIST, exp_id.encode())

        # Enforce capacity limit
        self._enforce_capacity()

        return exp_id

    def add_episode(
        self,
        experiences: List[bytes],
        final_rewards: bytes,
        model_version: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a complete episode with all experiences.

        This is more efficient than adding experiences one by one.

        Args:
            experiences: List of serialized experience bytes.
            final_rewards: Serialized final rewards for the episode.
            model_version: Model version used for this episode.
            metadata: Optional episode metadata (game_length, winner, etc.)

        Returns:
            Episode ID.

        Example:
            >>> exp_list = [serialize_experience(e) for e in episode_exps]
            >>> rewards = serialize_rewards(jnp.array([1.0, -1.0]))
            >>> episode_id = buffer.add_episode(exp_list, rewards, model_version=5)
        """
        episode_id = self._generate_id()
        timestamp = time.time()

        # Use pipeline for atomic batch operation
        pipe = self.redis.pipeline()

        exp_ids = []
        for i, exp_bytes in enumerate(experiences):
            exp_id = f"{episode_id}:{i}"
            exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

            pipe.hset(exp_key, mapping={
                b'data': exp_bytes,
                b'model_version': str(model_version).encode(),
                b'timestamp': str(timestamp).encode(),
                b'has_reward': b'1',
                b'episode_id': episode_id.encode(),
                b'final_rewards': final_rewards,
            })
            pipe.lpush(self.EXPERIENCE_LIST, exp_id.encode())
            exp_ids.append(exp_id)

        # Store episode metadata
        episode_key = f"{self.EPISODE_KEY}{episode_id}"
        episode_data = {
            b'num_experiences': str(len(experiences)).encode(),
            b'model_version': str(model_version).encode(),
            b'timestamp': str(timestamp).encode(),
        }
        if metadata:
            import msgpack
            episode_data[b'metadata'] = msgpack.packb(metadata)

        pipe.hset(episode_key, mapping=episode_data)
        pipe.lpush(self.EPISODE_LIST, episode_id.encode())

        pipe.execute()

        # Enforce capacity
        self._enforce_capacity()

        return episode_id

    def mark_episode_complete(
        self,
        episode_id: str,
        final_rewards: bytes,
    ) -> int:
        """Mark all experiences in an episode as complete with rewards.

        Args:
            episode_id: Episode ID to mark complete.
            final_rewards: Serialized final rewards.

        Returns:
            Number of experiences updated.
        """
        # Find all experiences for this episode
        pattern = f"{self.EXPERIENCE_KEY}{episode_id}:*"
        keys = list(self.redis.scan_iter(match=pattern))

        if not keys:
            return 0

        pipe = self.redis.pipeline()
        for key in keys:
            pipe.hset(key, mapping={
                b'has_reward': b'1',
                b'final_rewards': final_rewards,
            })
        pipe.execute()

        return len(keys)

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample_batch(
        self,
        batch_size: int,
        min_model_version: int = 0,
        require_rewards: bool = True,
    ) -> List[Dict[str, Any]]:
        """Sample a batch of experiences for training.

        Args:
            batch_size: Number of experiences to sample.
            min_model_version: Only sample experiences from this version or newer.
            require_rewards: Only sample experiences with final rewards.

        Returns:
            List of experience dicts with 'data', 'model_version', 'final_rewards'.

        Example:
            >>> batch = buffer.sample_batch(128)
            >>> jax_batch = batch_experiences_to_jax(batch)
        """
        # Get candidate experience IDs
        all_exp_ids = self.redis.lrange(self.EXPERIENCE_LIST, 0, -1)

        if not all_exp_ids:
            return []

        # Filter valid experiences
        valid_ids = []
        for exp_id_bytes in all_exp_ids:
            exp_id = exp_id_bytes.decode() if isinstance(exp_id_bytes, bytes) else exp_id_bytes
            exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

            # Check if experience meets criteria
            exp_meta = self.redis.hmget(exp_key, b'has_reward', b'model_version')

            if exp_meta[0] is None:
                continue

            has_reward = exp_meta[0] == b'1'
            model_ver = int(exp_meta[1]) if exp_meta[1] else 0

            if require_rewards and not has_reward:
                continue
            if model_ver < min_model_version:
                continue

            valid_ids.append(exp_id)

            # Early exit if we have enough candidates
            if len(valid_ids) >= batch_size * 3:
                break

        if not valid_ids:
            return []

        # Sample from valid experiences
        if len(valid_ids) <= batch_size:
            sampled_ids = valid_ids
        else:
            sampled_ids = list(np.random.choice(valid_ids, size=batch_size, replace=False))

        # Fetch full experience data
        batch = []
        pipe = self.redis.pipeline()
        for exp_id in sampled_ids:
            pipe.hgetall(f"{self.EXPERIENCE_KEY}{exp_id}")

        results = pipe.execute()

        for result in results:
            if result:
                batch.append({
                    'data': result.get(b'data', b''),
                    'model_version': int(result.get(b'model_version', b'0')),
                    'final_rewards': result.get(b'final_rewards', b''),
                })

        return batch

    def sample_recent(
        self,
        batch_size: int,
        max_age_seconds: float = 3600.0,
    ) -> List[Dict[str, Any]]:
        """Sample recent experiences (useful for prioritizing fresh data).

        Args:
            batch_size: Number of experiences to sample.
            max_age_seconds: Maximum age of experiences to consider.

        Returns:
            List of experience dicts.
        """
        cutoff_time = time.time() - max_age_seconds

        # Get recent experiences
        all_exp_ids = self.redis.lrange(self.EXPERIENCE_LIST, 0, batch_size * 5)

        valid_ids = []
        for exp_id_bytes in all_exp_ids:
            exp_id = exp_id_bytes.decode() if isinstance(exp_id_bytes, bytes) else exp_id_bytes
            exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

            exp_meta = self.redis.hmget(exp_key, b'timestamp', b'has_reward')

            if exp_meta[0] is None:
                continue

            timestamp = float(exp_meta[0])
            has_reward = exp_meta[1] == b'1'

            if timestamp >= cutoff_time and has_reward:
                valid_ids.append(exp_id)

        if not valid_ids:
            return []

        # Sample from recent valid experiences
        if len(valid_ids) <= batch_size:
            sampled_ids = valid_ids
        else:
            sampled_ids = list(np.random.choice(valid_ids, size=batch_size, replace=False))

        # Fetch data
        batch = []
        pipe = self.redis.pipeline()
        for exp_id in sampled_ids:
            pipe.hgetall(f"{self.EXPERIENCE_KEY}{exp_id}")

        results = pipe.execute()

        for result in results:
            if result:
                batch.append({
                    'data': result.get(b'data', b''),
                    'model_version': int(result.get(b'model_version', b'0')),
                    'final_rewards': result.get(b'final_rewards', b''),
                })

        return batch

    # =========================================================================
    # Capacity Management
    # =========================================================================

    def _enforce_capacity(self) -> None:
        """Remove old experiences when capacity is exceeded."""
        current_size = self.redis.llen(self.EXPERIENCE_LIST)

        if current_size > self.capacity:
            # Remove oldest experiences
            num_to_remove = current_size - self.capacity

            pipe = self.redis.pipeline()
            for _ in range(num_to_remove):
                # Pop from the end (oldest)
                exp_id_bytes = self.redis.rpop(self.EXPERIENCE_LIST)
                if exp_id_bytes:
                    exp_id = exp_id_bytes.decode() if isinstance(exp_id_bytes, bytes) else exp_id_bytes
                    pipe.delete(f"{self.EXPERIENCE_KEY}{exp_id}")
            pipe.execute()

    def trim_old_experiences(self, max_age_seconds: float) -> int:
        """Remove experiences older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of experiences removed.
        """
        cutoff_time = time.time() - max_age_seconds
        removed = 0

        # Scan from oldest
        all_exp_ids = self.redis.lrange(self.EXPERIENCE_LIST, -1000, -1)  # Oldest 1000

        pipe = self.redis.pipeline()
        ids_to_remove = []

        for exp_id_bytes in all_exp_ids:
            exp_id = exp_id_bytes.decode() if isinstance(exp_id_bytes, bytes) else exp_id_bytes
            exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

            timestamp = self.redis.hget(exp_key, b'timestamp')
            if timestamp and float(timestamp) < cutoff_time:
                ids_to_remove.append(exp_id)
                pipe.delete(exp_key)
                pipe.lrem(self.EXPERIENCE_LIST, 1, exp_id_bytes)

        if ids_to_remove:
            pipe.execute()
            removed = len(ids_to_remove)

        return removed

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> BufferStats:
        """Get buffer statistics.

        Returns:
            BufferStats with current buffer state.
        """
        total_experiences = self.redis.llen(self.EXPERIENCE_LIST)
        total_episodes = self.redis.llen(self.EPISODE_LIST)

        # Sample to estimate reward percentage
        sample_ids = self.redis.lrange(self.EXPERIENCE_LIST, 0, 100)
        with_rewards = 0
        oldest_ts = None
        newest_ts = None

        for exp_id_bytes in sample_ids:
            exp_id = exp_id_bytes.decode() if isinstance(exp_id_bytes, bytes) else exp_id_bytes
            exp_key = f"{self.EXPERIENCE_KEY}{exp_id}"

            exp_meta = self.redis.hmget(exp_key, b'has_reward', b'timestamp')

            if exp_meta[0] == b'1':
                with_rewards += 1

            if exp_meta[1]:
                ts = float(exp_meta[1])
                if oldest_ts is None or ts < oldest_ts:
                    oldest_ts = ts
                if newest_ts is None or ts > newest_ts:
                    newest_ts = ts

        rewards_pct = (with_rewards / max(len(sample_ids), 1)) * 100

        return BufferStats(
            total_experiences=total_experiences,
            total_episodes=total_episodes,
            capacity=self.capacity,
            fullness_pct=(total_experiences / self.capacity) * 100,
            oldest_timestamp=oldest_ts,
            newest_timestamp=newest_ts,
            experiences_with_rewards_pct=rewards_pct,
        )

    def get_size(self) -> int:
        """Get number of experiences in buffer."""
        return self.redis.llen(self.EXPERIENCE_LIST)

    def is_ready_for_training(self, min_experiences: int = 1000) -> bool:
        """Check if buffer has enough experiences for training.

        Args:
            min_experiences: Minimum number of experiences required.

        Returns:
            True if buffer is ready for training.
        """
        return self.get_size() >= min_experiences

    # =========================================================================
    # Cleanup
    # =========================================================================

    def clear(self) -> None:
        """Clear all data from the buffer."""
        # Get all keys with our prefix
        patterns = [
            f"{self.EXPERIENCE_KEY}*",
            f"{self.EPISODE_KEY}*",
        ]

        for pattern in patterns:
            keys = list(self.redis.scan_iter(match=pattern))
            if keys:
                self.redis.delete(*keys)

        self.redis.delete(self.EXPERIENCE_LIST)
        self.redis.delete(self.EPISODE_LIST)
        self.redis.delete(self.METADATA_KEY)

    def close(self) -> None:
        """Close Redis connection."""
        self.redis.close()

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> 'RedisReplayBuffer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
