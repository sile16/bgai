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
    avg_surprise_score: Optional[float] = None


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
    SURPRISE_SORTED_SET = "bgai:buffer:surprise"  # Sorted set for surprise-weighted sampling
    TOTAL_GAMES_KEY = "bgai:buffer:total_games"  # Monotonic counter for total games added

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
        surprise_score: Optional[float] = None,
    ) -> str:
        """Add a complete episode with all experiences.

        This is more efficient than adding experiences one by one.

        Args:
            experiences: List of serialized experience bytes.
            final_rewards: Serialized final rewards for the episode.
            model_version: Model version used for this episode.
            metadata: Optional episode metadata (game_length, winner, etc.)
            surprise_score: Optional surprise score for prioritized sampling.
                Higher scores indicate more "surprising" games where the outcome
                differed significantly from value predictions.

        Returns:
            Episode ID.

        Example:
            >>> exp_list = [serialize_experience(e) for e in episode_exps]
            >>> rewards = serialize_rewards(jnp.array([1.0, -1.0]))
            >>> episode_id = buffer.add_episode(exp_list, rewards, model_version=5, surprise_score=0.8)
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
        if surprise_score is not None:
            episode_data[b'surprise_score'] = str(surprise_score).encode()
        if metadata:
            import msgpack
            episode_data[b'metadata'] = msgpack.packb(metadata)

        pipe.hset(episode_key, mapping=episode_data)
        pipe.lpush(self.EPISODE_LIST, episode_id.encode())

        # Add to surprise sorted set for weighted sampling
        # Use surprise_score as the score (higher = more interesting)
        # Add small random jitter to break ties
        score = (surprise_score or 0.0) + np.random.uniform(0, 0.001)
        pipe.zadd(self.SURPRISE_SORTED_SET, {episode_id: score})

        # Increment monotonic game counter (never decreases even after eviction)
        pipe.incr(self.TOTAL_GAMES_KEY)

        pipe.execute()

        # Enforce episode capacity only - this properly cleans up experiences,
        # episode metadata, and surprise entries atomically.
        # Note: We don't use _enforce_capacity() for individual experiences
        # because it leaves orphaned episode metadata and surprise entries.
        self._enforce_episode_capacity()

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
        # Get total buffer size
        total_size = self.redis.llen(self.EXPERIENCE_LIST)
        if total_size == 0:
            return []

        # Sample from a window at the start (most recent experiences)
        # Old experiences may be orphaned (evicted data but metadata remains)
        # Use larger window to increase variety while preferring recent data
        sample_window = min(batch_size * 20, total_size)  # Window of 20x batch size
        offset = 0  # Start from most recent

        # Get candidate experience IDs from the window
        candidate_ids = self.redis.lrange(
            self.EXPERIENCE_LIST, offset, offset + sample_window - 1
        )

        if not candidate_ids:
            return []

        # Randomly shuffle candidates to avoid bias
        candidate_ids = list(candidate_ids)
        np.random.shuffle(candidate_ids)

        # Filter valid experiences (with early exit)
        valid_ids = []
        for exp_id_bytes in candidate_ids:
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
            if len(valid_ids) >= batch_size * 2:
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

    def sample_batch_surprise_weighted(
        self,
        batch_size: int,
        surprise_weight: float = 0.5,
        min_model_version: int = 0,
    ) -> List[Dict[str, Any]]:
        """Sample a batch with surprise-weighted episode selection.

        Episodes with higher surprise scores (where outcome differed from
        predictions) are sampled more frequently.

        Args:
            batch_size: Number of experiences to sample.
            surprise_weight: Weight for surprise-based sampling (0-1).
                0 = uniform sampling, 1 = fully surprise-weighted.
            min_model_version: Only sample from this version or newer.

        Returns:
            List of experience dicts.
        """
        # Get episodes with high surprise scores
        # ZREVRANGE returns highest scores first
        num_episodes_to_consider = min(100, self.redis.zcard(self.SURPRISE_SORTED_SET))
        if num_episodes_to_consider == 0:
            return self.sample_batch(batch_size, min_model_version)

        # Get top surprising episodes
        high_surprise_episodes = self.redis.zrevrange(
            self.SURPRISE_SORTED_SET, 0, num_episodes_to_consider - 1, withscores=True
        )

        if not high_surprise_episodes:
            return self.sample_batch(batch_size, min_model_version)

        # Filter by model version and build weighted list using pipeline
        # First, fetch all model versions in a single pipeline
        pipe = self.redis.pipeline()
        episode_ids = []
        episode_scores = []
        for ep_id_bytes, score in high_surprise_episodes:
            ep_id = ep_id_bytes.decode() if isinstance(ep_id_bytes, bytes) else ep_id_bytes
            episode_key = f"{self.EPISODE_KEY}{ep_id}"
            pipe.hget(episode_key, b'model_version')
            episode_ids.append(ep_id)
            episode_scores.append(score)

        model_versions = pipe.execute()

        valid_episodes = []
        weights = []
        for ep_id, score, model_ver_bytes in zip(episode_ids, episode_scores, model_versions):
            if model_ver_bytes:
                model_ver = int(model_ver_bytes)
                if model_ver >= min_model_version:
                    valid_episodes.append(ep_id)
                    # Blend uniform and surprise-weighted
                    # Higher score = more likely to sample
                    blended_weight = (1 - surprise_weight) + surprise_weight * (score + 0.1)
                    weights.append(blended_weight)

        if not valid_episodes:
            return self.sample_batch(batch_size, min_model_version)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample episodes weighted by surprise
        num_episodes = min(len(valid_episodes), batch_size // 10 + 1)  # Sample from multiple episodes
        sampled_episode_indices = np.random.choice(
            len(valid_episodes), size=num_episodes, replace=True, p=weights
        )
        sampled_episodes = [valid_episodes[i] for i in sampled_episode_indices]

        # Get experiences from sampled episodes using pipeline
        # Get unique episodes to avoid duplicate lookups
        unique_episodes = list(set(sampled_episodes))
        pipe = self.redis.pipeline()
        for ep_id in unique_episodes:
            episode_key = f"{self.EPISODE_KEY}{ep_id}"
            pipe.hget(episode_key, b'num_experiences')

        num_experiences_results = pipe.execute()

        # Build mapping from episode to num_experiences
        ep_to_num_exp = {}
        for ep_id, num_exp_bytes in zip(unique_episodes, num_experiences_results):
            if num_exp_bytes:
                ep_to_num_exp[ep_id] = int(num_exp_bytes)

        # Build experience IDs from sampled episodes
        all_exp_ids = []
        for ep_id in sampled_episodes:
            if ep_id in ep_to_num_exp:
                num_exp = ep_to_num_exp[ep_id]
                for i in range(num_exp):
                    all_exp_ids.append(f"{ep_id}:{i}")

        if not all_exp_ids:
            return self.sample_batch(batch_size, min_model_version)

        # Sample experiences from the collected episode experiences
        if len(all_exp_ids) <= batch_size:
            sampled_ids = all_exp_ids
        else:
            sampled_ids = list(np.random.choice(all_exp_ids, size=batch_size, replace=False))

        # Fetch experience data
        batch = []
        pipe = self.redis.pipeline()
        for exp_id in sampled_ids:
            pipe.hgetall(f"{self.EXPERIENCE_KEY}{exp_id}")

        results = pipe.execute()

        for result in results:
            if result and result.get(b'data'):
                batch.append({
                    'data': result.get(b'data', b''),
                    'model_version': int(result.get(b'model_version', b'0')),
                    'final_rewards': result.get(b'final_rewards', b''),
                })

        # If surprise-weighted sampling didn't get enough samples (orphaned episodes),
        # fall back to uniform sampling from recent experiences
        if len(batch) < batch_size:
            fallback_batch = self.sample_batch(batch_size, min_model_version)
            if fallback_batch:
                return fallback_batch

        return batch

    # =========================================================================
    # Capacity Management
    # =========================================================================

    def _enforce_capacity(self) -> None:
        """DEPRECATED: Individual experience eviction causes orphaned metadata.

        This method is kept for backwards compatibility but does nothing.
        Use _enforce_episode_capacity() instead, which atomically removes
        complete episodes including all experiences, metadata, and surprise entries.
        """
        # No-op: Episode-based eviction is now the only mechanism.
        # Individual experience eviction was removed because it left
        # orphaned episode metadata and surprise sorted set entries,
        # causing sampling to reference non-existent experiences.
        pass

    def _enforce_episode_capacity(self) -> None:
        """Remove old episodes when episode capacity is exceeded (FIFO)."""
        current_episodes = self.redis.llen(self.EPISODE_LIST)

        if current_episodes > self.episode_capacity:
            num_to_remove = current_episodes - self.episode_capacity

            for _ in range(num_to_remove):
                # Pop oldest episode from the end
                episode_id_bytes = self.redis.rpop(self.EPISODE_LIST)
                if episode_id_bytes:
                    episode_id = episode_id_bytes.decode() if isinstance(episode_id_bytes, bytes) else episode_id_bytes
                    self._delete_episode(episode_id)

    def _delete_episode(self, episode_id: str) -> None:
        """Delete an episode and all its experiences."""
        pipe = self.redis.pipeline()

        # Delete episode metadata
        pipe.delete(f"{self.EPISODE_KEY}{episode_id}")

        # Remove from surprise sorted set
        pipe.zrem(self.SURPRISE_SORTED_SET, episode_id)

        # Find and delete all experiences for this episode
        pattern = f"{self.EXPERIENCE_KEY}{episode_id}:*"
        exp_keys = list(self.redis.scan_iter(match=pattern, count=1000))
        for key in exp_keys:
            pipe.delete(key)
            # Also remove from experience list
            exp_id = key.decode().replace(self.EXPERIENCE_KEY, "")
            pipe.lrem(self.EXPERIENCE_LIST, 0, exp_id.encode())

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

    def get_total_games(self) -> int:
        """Get total number of games ever added (monotonic counter).

        Unlike get_episode_count() which returns current buffer size,
        this returns the total count of games ever added, which only
        increases and is suitable for triggering training batches.
        """
        count = self.redis.get(self.TOTAL_GAMES_KEY)
        return int(count) if count else 0

    def get_episode_count(self) -> int:
        """Get current number of episodes in buffer."""
        return self.redis.llen(self.EPISODE_LIST)

    def get_surprise_stats(self) -> Dict[str, Any]:
        """Get statistics about surprise scores in the buffer.

        Returns:
            Dict with surprise score statistics.
        """
        # Get count of episodes with surprise scores
        num_episodes = self.redis.zcard(self.SURPRISE_SORTED_SET)

        if num_episodes == 0:
            return {
                'count': 0,
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
            }

        # Get top and bottom scores
        top_scores = self.redis.zrevrange(
            self.SURPRISE_SORTED_SET, 0, 0, withscores=True
        )
        bottom_scores = self.redis.zrange(
            self.SURPRISE_SORTED_SET, 0, 0, withscores=True
        )

        max_score = top_scores[0][1] if top_scores else 0.0
        min_score = bottom_scores[0][1] if bottom_scores else 0.0

        # Sample to estimate mean (for efficiency, don't scan all)
        sample_size = min(num_episodes, 100)
        sample_scores = self.redis.zrevrange(
            self.SURPRISE_SORTED_SET, 0, sample_size - 1, withscores=True
        )
        mean_score = sum(s[1] for s in sample_scores) / len(sample_scores) if sample_scores else 0.0

        return {
            'count': num_episodes,
            'max': max_score,
            'min': min_score,
            'mean': mean_score,
        }

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

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()
