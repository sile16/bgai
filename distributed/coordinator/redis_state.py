"""Redis state management for distributed training.

This module provides utilities for managing cluster state in Redis,
replacing Ray's actor-based coordination with a simpler Redis-only approach.

Key Schema:
-----------
bgai:model:version          - int: current model version
bgai:model:weights          - bytes: pickled model parameters
bgai:model:config           - json: training config
bgai:run:id                 - string: current MLflow run ID
bgai:run:status             - string: running|paused|stopped

bgai:workers:{worker_id}    - json: worker info (with TTL)
bgai:workers:list           - set: all worker IDs

bgai:metrics:targets        - json: Prometheus scrape targets
"""

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import redis


# =============================================================================
# Constants
# =============================================================================

# Key prefixes
PREFIX = "bgai"
KEY_MODEL_VERSION = f"{PREFIX}:model:version"
KEY_MODEL_WEIGHTS = f"{PREFIX}:model:weights"
KEY_MODEL_CONFIG = f"{PREFIX}:model:config"
KEY_RUN_ID = f"{PREFIX}:run:id"
KEY_RUN_STATUS = f"{PREFIX}:run:status"
KEY_WORKERS_LIST = f"{PREFIX}:workers:list"
KEY_METRICS_TARGETS = f"{PREFIX}:metrics:targets"
KEY_WARM_TREE = f"{PREFIX}:model:warm_tree"
KEY_WARM_TREE_VERSION = f"{PREFIX}:model:warm_tree_version"

# TTL values (seconds)
WORKER_TTL = 60  # Workers expire after 60s without heartbeat
HEARTBEAT_INTERVAL = 10  # Recommended heartbeat interval


def worker_key(worker_id: str) -> str:
    """Get Redis key for a worker."""
    return f"{PREFIX}:workers:{worker_id}"


# =============================================================================
# Data Classes
# =============================================================================

class WorkerStatus(Enum):
    """Worker status states."""
    IDLE = "idle"
    WORKING = "working"
    DISCONNECTED = "disconnected"


class RunStatus(Enum):
    """Training run status."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class WorkerInfo:
    """Information about a registered worker."""
    worker_id: str
    worker_type: str  # 'game', 'training', 'eval'
    device_type: str  # 'cuda', 'metal', 'cpu'
    device_name: str
    hostname: str
    metrics_port: int
    status: str = "idle"
    games_generated: int = 0
    training_steps: int = 0
    model_version: int = 0
    registered_at: float = 0.0
    last_heartbeat: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerInfo':
        """Create from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> 'WorkerInfo':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))


# =============================================================================
# Redis State Manager
# =============================================================================

class RedisStateManager:
    """Manages cluster state in Redis.

    This class provides a high-level interface for all Redis operations
    needed by the distributed training system.

    Example:
        >>> state = RedisStateManager(host='localhost', port=6379)
        >>> state.set_model_version(1)
        >>> version = state.get_model_version()
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
    ):
        """Initialize Redis connection.

        Args:
            host: Redis server hostname.
            port: Redis server port.
            password: Redis password (optional).
            db: Redis database number.
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=False,  # We handle encoding ourselves
        )
        self._host = host
        self._port = port

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self.redis.ping()
        except redis.ConnectionError:
            return False

    # =========================================================================
    # Model Weights
    # =========================================================================

    def get_model_version(self) -> int:
        """Get current model version."""
        version = self.redis.get(KEY_MODEL_VERSION)
        return int(version) if version else 0

    def set_model_version(self, version: int) -> None:
        """Set current model version."""
        self.redis.set(KEY_MODEL_VERSION, str(version))

    def get_model_weights(self) -> Optional[bytes]:
        """Get current model weights (serialized)."""
        return self.redis.get(KEY_MODEL_WEIGHTS)

    def set_model_weights(self, weights: bytes, version: int) -> bool:
        """Set model weights atomically with version.

        Only updates if version is newer than current.

        Args:
            weights: Serialized model weights.
            version: New model version.

        Returns:
            True if updated, False if version was stale.
        """
        # Use transaction to ensure atomicity
        pipe = self.redis.pipeline()
        try:
            # Watch the version key
            pipe.watch(KEY_MODEL_VERSION)
            current_version = pipe.get(KEY_MODEL_VERSION)
            current_version = int(current_version) if current_version else 0

            if version <= current_version:
                pipe.unwatch()
                return False

            # Execute update atomically
            pipe.multi()
            pipe.set(KEY_MODEL_WEIGHTS, weights)
            pipe.set(KEY_MODEL_VERSION, str(version))
            pipe.execute()
            return True

        except redis.WatchError:
            # Another client modified the version
            return False

    def get_model_weights_if_newer(self, current_version: int) -> Optional[tuple]:
        """Get model weights if newer version available.

        Args:
            current_version: Worker's current model version.

        Returns:
            Tuple of (weights, version) if newer available, None otherwise.
        """
        version = self.get_model_version()
        if version > current_version:
            weights = self.get_model_weights()
            if weights:
                return (weights, version)
        return None

    # =========================================================================
    # Warm Tree (pre-computed MCTS state)
    # =========================================================================

    def get_warm_tree(self) -> Optional[bytes]:
        """Get current warm tree (serialized MCTS state)."""
        return self.redis.get(KEY_WARM_TREE)

    def get_warm_tree_version(self) -> int:
        """Get model version associated with warm tree."""
        version = self.redis.get(KEY_WARM_TREE_VERSION)
        return int(version) if version else 0

    def set_warm_tree(self, tree_data: bytes, version: int) -> None:
        """Set warm tree with associated model version.

        Args:
            tree_data: Serialized MCTS tree state.
            version: Model version this tree was built from.
        """
        pipe = self.redis.pipeline()
        pipe.set(KEY_WARM_TREE, tree_data)
        pipe.set(KEY_WARM_TREE_VERSION, str(version))
        pipe.execute()

    def get_warm_tree_if_newer(self, current_version: int) -> Optional[tuple]:
        """Get warm tree if newer version available.

        Args:
            current_version: Worker's current warm tree version.

        Returns:
            Tuple of (tree_data, version) if newer available, None otherwise.
        """
        version = self.get_warm_tree_version()
        if version > current_version:
            tree_data = self.get_warm_tree()
            if tree_data:
                return (tree_data, version)
        return None

    def delete_warm_tree(self) -> None:
        """Delete the warm tree from Redis."""
        self.redis.delete(KEY_WARM_TREE)
        self.redis.delete(KEY_WARM_TREE_VERSION)

    # =========================================================================
    # Training Run
    # =========================================================================

    def get_run_id(self) -> Optional[str]:
        """Get current MLflow run ID."""
        run_id = self.redis.get(KEY_RUN_ID)
        return run_id.decode() if run_id else None

    def set_run_id(self, run_id: str) -> None:
        """Set current MLflow run ID."""
        self.redis.set(KEY_RUN_ID, run_id)

    def get_run_status(self) -> str:
        """Get current training run status."""
        status = self.redis.get(KEY_RUN_STATUS)
        return status.decode() if status else RunStatus.STOPPED.value

    def set_run_status(self, status: Union[str, RunStatus]) -> None:
        """Set training run status."""
        if isinstance(status, RunStatus):
            status = status.value
        self.redis.set(KEY_RUN_STATUS, status)

    def is_training_active(self) -> bool:
        """Check if training is currently active."""
        status = self.redis.get(KEY_RUN_STATUS)
        if status is None:
            # Default to active when unset so fresh clusters train
            return True
        return status.decode() == RunStatus.RUNNING.value

    # =========================================================================
    # Worker Registry
    # =========================================================================

    def register_worker(self, info: WorkerInfo, ttl: int = WORKER_TTL) -> None:
        """Register a worker with TTL.

        Args:
            info: Worker information.
            ttl: Time-to-live in seconds.
        """
        info.registered_at = time.time()
        info.last_heartbeat = time.time()

        key = worker_key(info.worker_id)
        self.redis.setex(key, ttl, info.to_json())
        self.redis.sadd(KEY_WORKERS_LIST, info.worker_id)

    def heartbeat_worker(
        self,
        worker_id: str,
        stats: Optional[Dict[str, Any]] = None,
        ttl: int = WORKER_TTL,
    ) -> bool:
        """Update worker heartbeat and stats.

        Args:
            worker_id: Worker ID.
            stats: Optional stats to update (games_generated, training_steps, status).
            ttl: Time-to-live in seconds.

        Returns:
            True if worker exists and was updated, False otherwise.
        """
        key = worker_key(worker_id)
        data = self.redis.get(key)

        if not data:
            return False

        info = WorkerInfo.from_json(data.decode())
        info.last_heartbeat = time.time()

        if stats:
            if 'games_generated' in stats:
                info.games_generated += stats['games_generated']
            if 'training_steps' in stats:
                info.training_steps += stats['training_steps']
            if 'status' in stats:
                info.status = stats['status']
            if 'model_version' in stats:
                info.model_version = stats['model_version']

        self.redis.setex(key, ttl, info.to_json())
        return True

    def deregister_worker(self, worker_id: str) -> bool:
        """Remove a worker from registry.

        Args:
            worker_id: Worker ID.

        Returns:
            True if worker was removed, False if not found.
        """
        key = worker_key(worker_id)
        removed = self.redis.delete(key)
        self.redis.srem(KEY_WORKERS_LIST, worker_id)
        return removed > 0

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker information.

        Args:
            worker_id: Worker ID.

        Returns:
            WorkerInfo or None if not found.
        """
        key = worker_key(worker_id)
        data = self.redis.get(key)
        if data:
            return WorkerInfo.from_json(data.decode())
        return None

    def get_all_workers(self) -> List[WorkerInfo]:
        """Get all active workers.

        Returns:
            List of WorkerInfo for all workers with valid TTL.
        """
        workers = []
        worker_ids = self.redis.smembers(KEY_WORKERS_LIST)

        for wid in worker_ids:
            wid_str = wid.decode() if isinstance(wid, bytes) else wid
            info = self.get_worker(wid_str)
            if info:
                workers.append(info)
            else:
                # Worker expired, remove from set
                self.redis.srem(KEY_WORKERS_LIST, wid)

        return workers

    def get_workers_by_type(self, worker_type: str) -> List[WorkerInfo]:
        """Get workers of a specific type.

        Args:
            worker_type: Worker type ('game', 'training', 'eval').

        Returns:
            List of WorkerInfo matching the type.
        """
        return [w for w in self.get_all_workers() if w.worker_type == worker_type]

    def get_worker_counts(self) -> Dict[str, int]:
        """Get count of workers by type.

        Returns:
            Dict with counts by worker type.
        """
        workers = self.get_all_workers()
        counts = {'game': 0, 'training': 0, 'eval': 0, 'total': 0}

        for w in workers:
            counts[w.worker_type] = counts.get(w.worker_type, 0) + 1
            counts['total'] += 1

        return counts

    # =========================================================================
    # Metrics Discovery
    # =========================================================================

    def update_metrics_targets(self) -> None:
        """Update Prometheus scrape targets from worker registry.

        Generates a targets list in Prometheus file_sd format.
        """
        workers = self.get_all_workers()
        targets = []

        for w in workers:
            targets.append({
                'targets': [f'{w.hostname}:{w.metrics_port}'],
                'labels': {
                    'worker_id': w.worker_id,
                    'worker_type': w.worker_type,
                    'device_type': w.device_type,
                    'hostname': w.hostname,
                }
            })

        self.redis.set(KEY_METRICS_TARGETS, json.dumps(targets))

    def get_metrics_targets(self) -> List[Dict[str, Any]]:
        """Get Prometheus scrape targets.

        Returns:
            List of target configs in file_sd format.
        """
        data = self.redis.get(KEY_METRICS_TARGETS)
        if data:
            return json.loads(data.decode())
        return []

    # =========================================================================
    # Cluster Status
    # =========================================================================

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status.

        Returns:
            Dict with cluster status information.
        """
        workers = self.get_all_workers()

        total_games = sum(w.games_generated for w in workers)
        total_steps = sum(w.training_steps for w in workers)

        game_workers = [w for w in workers if w.worker_type == 'game']
        training_workers = [w for w in workers if w.worker_type == 'training']
        eval_workers = [w for w in workers if w.worker_type == 'eval']

        return {
            'model_version': self.get_model_version(),
            'run_id': self.get_run_id(),
            'run_status': self.get_run_status(),
            'workers': {
                'game': len(game_workers),
                'training': len(training_workers),
                'eval': len(eval_workers),
                'total': len(workers),
            },
            'active_workers': [w.to_dict() for w in workers],
            'total_games_generated': total_games,
            'total_training_steps': total_steps,
            'timestamp': time.time(),
        }

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_stale_workers(self) -> int:
        """Remove workers from list that have expired.

        Returns:
            Number of stale workers cleaned up.
        """
        cleaned = 0
        worker_ids = self.redis.smembers(KEY_WORKERS_LIST)

        for wid in worker_ids:
            wid_str = wid.decode() if isinstance(wid, bytes) else wid
            key = worker_key(wid_str)
            if not self.redis.exists(key):
                self.redis.srem(KEY_WORKERS_LIST, wid)
                cleaned += 1

        return cleaned

    def reset_cluster_state(self) -> None:
        """Reset all cluster state (for testing/cleanup).

        WARNING: This removes all workers and resets model version.
        """
        # Get all keys with our prefix
        keys = self.redis.keys(f"{PREFIX}:*")
        if keys:
            self.redis.delete(*keys)


# =============================================================================
# Utility Functions
# =============================================================================

def create_state_manager(config: Dict[str, Any]) -> RedisStateManager:
    """Create a RedisStateManager from config dict.

    Args:
        config: Config with redis_host, redis_port, redis_password.

    Returns:
        Configured RedisStateManager.
    """
    return RedisStateManager(
        host=config.get('redis_host', 'localhost'),
        port=config.get('redis_port', 6379),
        password=config.get('redis_password'),
    )
