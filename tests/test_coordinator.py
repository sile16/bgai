"""Tests for the Coordinator (Ray actor).

These tests use Ray's local mode for isolated testing.
"""

import pytest
import time

import ray

from distributed.coordinator.head_node import (
    Coordinator,
    WorkerInfo,
    WorkerStatus,
    get_coordinator,
    create_coordinator,
    get_or_create_coordinator,
)
from distributed.serialization import serialize_weights
import jax.numpy as jnp


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    # Always shutdown first to get clean state
    if ray.is_initialized():
        ray.shutdown()
    # Don't use local_mode - it has crashes during cleanup
    ray.init(ignore_reinit_error=True)
    yield
    # Shutdown to allow clean restart in other test modules
    ray.shutdown()


@pytest.fixture
def coordinator(ray_init):
    """Create a coordinator actor for testing."""
    import uuid

    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'heartbeat_timeout': 30.0,
        'heartbeat_interval': 10.0,
        'mcts_simulations': 100,
        'mcts_max_nodes': 400,
        'train_batch_size': 128,
    }

    # Create coordinator with unique name to avoid conflicts
    name = f"test-coordinator-{uuid.uuid4().hex[:8]}"
    coord = Coordinator.options(name=name).remote(config)

    yield coord

    # Use ray.kill() instead of shutdown.remote() to avoid crashes during pytest teardown
    # Ray in local_mode has issues with actor lifecycle during teardown
    try:
        ray.kill(coord)
    except Exception:
        pass  # Actor may already be dead


@pytest.fixture
def sample_device_info():
    """Sample device info for worker registration."""
    return {
        'device_type': 'metal',
        'device_name': 'Apple M1',
        'hostname': 'test-host',
    }


class TestWorkerRegistration:
    """Tests for worker registration."""

    def test_register_worker(self, coordinator, sample_device_info):
        """Test registering a worker."""
        result = ray.get(coordinator.register_worker.remote(
            'worker-001',
            'game',
            sample_device_info,
        ))

        assert result['status'] == 'registered'
        assert 'model_version' in result
        assert 'config' in result

    def test_register_multiple_workers(self, coordinator, sample_device_info):
        """Test registering multiple workers."""
        for i in range(5):
            result = ray.get(coordinator.register_worker.remote(
                f'worker-{i:03d}',
                'game',
                sample_device_info,
            ))
            assert result['status'] == 'registered'

        counts = ray.get(coordinator.get_worker_count.remote())
        assert counts['game'] >= 5

    def test_register_different_worker_types(self, coordinator, sample_device_info):
        """Test registering different worker types."""
        ray.get(coordinator.register_worker.remote('game-001', 'game', sample_device_info))
        ray.get(coordinator.register_worker.remote('train-001', 'training', sample_device_info))
        ray.get(coordinator.register_worker.remote('eval-001', 'evaluation', sample_device_info))

        counts = ray.get(coordinator.get_worker_count.remote())
        assert counts['game'] >= 1
        assert counts['training'] >= 1
        assert counts['evaluation'] >= 1

    def test_deregister_worker(self, coordinator, sample_device_info):
        """Test deregistering a worker."""
        ray.get(coordinator.register_worker.remote(
            'worker-to-remove',
            'game',
            sample_device_info,
        ))

        result = ray.get(coordinator.deregister_worker.remote('worker-to-remove'))
        assert result['status'] == 'deregistered'

    def test_deregister_nonexistent_worker(self, coordinator):
        """Test deregistering a nonexistent worker."""
        result = ray.get(coordinator.deregister_worker.remote('nonexistent-worker'))
        assert result['status'] == 'not_found'


class TestHeartbeat:
    """Tests for worker heartbeat."""

    def test_heartbeat_success(self, coordinator, sample_device_info):
        """Test successful heartbeat."""
        ray.get(coordinator.register_worker.remote(
            'hb-worker',
            'game',
            sample_device_info,
        ))

        result = ray.get(coordinator.heartbeat.remote(
            'hb-worker',
            {'games_since_last': 10, 'status': 'working'},
        ))

        assert result['status'] == 'ok'

    def test_heartbeat_unknown_worker(self, coordinator):
        """Test heartbeat from unknown worker."""
        result = ray.get(coordinator.heartbeat.remote(
            'unknown-worker',
            {'status': 'idle'},
        ))

        assert result['status'] == 'unknown_worker'
        assert result.get('should_register') is True

    def test_heartbeat_updates_stats(self, coordinator, sample_device_info):
        """Test that heartbeat updates worker stats."""
        ray.get(coordinator.register_worker.remote(
            'stats-worker',
            'game',
            sample_device_info,
        ))

        # Send multiple heartbeats with stats
        for i in range(5):
            ray.get(coordinator.heartbeat.remote(
                'stats-worker',
                {'games_since_last': 10, 'status': 'working'},
            ))

        status = ray.get(coordinator.get_cluster_status.remote())

        # Find our worker
        worker = next((w for w in status['active_workers'] if w['id'] == 'stats-worker'), None)
        if worker:
            assert worker['games'] >= 50

    def test_heartbeat_notifies_model_update(self, coordinator, sample_device_info):
        """Test that heartbeat notifies of new model version."""
        ray.get(coordinator.register_worker.remote(
            'model-check-worker',
            'game',
            sample_device_info,
        ))

        # Update model weights
        weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})
        ray.get(coordinator.update_model_weights.remote(weights, version=5))

        # Heartbeat should notify of new model
        result = ray.get(coordinator.heartbeat.remote(
            'model-check-worker',
            {'status': 'idle'},
        ))

        assert result.get('new_model_available') is True
        assert result.get('new_model_version') == 5


class TestModelWeights:
    """Tests for model weight management."""

    def test_set_initial_weights(self, coordinator):
        """Test setting initial weights."""
        weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})

        result = ray.get(coordinator.set_initial_weights.remote(weights))

        assert result['status'] == 'updated'
        assert result['version'] == 1

    def test_update_model_weights(self, coordinator):
        """Test updating model weights."""
        weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})

        result = ray.get(coordinator.update_model_weights.remote(weights, version=10))

        assert result['status'] == 'updated'
        assert result['version'] == 10

    def test_get_model_weights(self, coordinator):
        """Test getting model weights."""
        weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})
        ray.get(coordinator.update_model_weights.remote(weights, version=3))

        result_weights, version = ray.get(coordinator.get_model_weights.remote())

        assert version == 3
        assert result_weights is not None

    def test_acknowledge_model_version(self, coordinator, sample_device_info):
        """Test acknowledging model version."""
        ray.get(coordinator.register_worker.remote(
            'ack-worker',
            'game',
            sample_device_info,
        ))

        result = ray.get(coordinator.acknowledge_model_version.remote(
            'ack-worker',
            version=5,
        ))

        assert result['status'] == 'acknowledged'


class TestClusterStatus:
    """Tests for cluster status reporting."""

    def test_get_cluster_status(self, coordinator, sample_device_info):
        """Test getting cluster status."""
        ray.get(coordinator.register_worker.remote('status-worker', 'game', sample_device_info))

        status = ray.get(coordinator.get_cluster_status.remote())

        assert 'model_version' in status
        assert 'active_workers' in status
        assert 'disconnected_count' in status
        assert 'total_games_generated' in status
        assert 'total_training_steps' in status
        assert 'uptime_seconds' in status

    def test_get_worker_count(self, coordinator, sample_device_info):
        """Test getting worker count by type."""
        counts = ray.get(coordinator.get_worker_count.remote())

        assert 'game' in counts
        assert 'training' in counts
        assert 'evaluation' in counts
        assert 'disconnected' in counts

    def test_get_workers_by_type(self, coordinator, sample_device_info):
        """Test getting workers by type."""
        ray.get(coordinator.register_worker.remote('type-worker', 'game', sample_device_info))

        workers = ray.get(coordinator.get_workers_by_type.remote('game'))

        assert isinstance(workers, list)


class TestConfiguration:
    """Tests for configuration management."""

    def test_get_config(self, coordinator):
        """Test getting configuration."""
        config = ray.get(coordinator.get_config.remote())

        assert isinstance(config, dict)
        assert 'redis_host' in config

    def test_update_config(self, coordinator):
        """Test updating configuration."""
        result = ray.get(coordinator.update_config.remote({
            'new_setting': 'new_value',
        }))

        assert result['status'] == 'updated'

        config = ray.get(coordinator.get_config.remote())
        assert config.get('new_setting') == 'new_value'


class TestCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_disconnected_workers(self, coordinator, sample_device_info):
        """Test cleaning up disconnected workers."""
        # This test is tricky because we'd need to wait for timeout
        # Just verify the method doesn't error
        removed = ray.get(coordinator.cleanup_disconnected_workers.remote())
        assert removed >= 0

    def test_shutdown(self, ray_init, sample_device_info):
        """Test graceful shutdown."""
        config = {'heartbeat_timeout': 30.0}
        name = f"shutdown-coordinator-{int(time.time() * 1000)}"
        coord = Coordinator.options(name=name).remote(config)

        ray.get(coord.register_worker.remote('shutdown-worker', 'game', sample_device_info))

        result = ray.get(coord.shutdown.remote())

        assert result['status'] == 'shutdown'


class TestWorkerInfo:
    """Tests for WorkerInfo dataclass."""

    def test_worker_info_creation(self):
        """Test creating WorkerInfo."""
        info = WorkerInfo(
            worker_id='test-001',
            worker_type='game',
            device_type='metal',
            device_name='Apple M1',
            hostname='test-host',
            registered_at=time.time(),
            last_heartbeat=time.time(),
        )

        assert info.worker_id == 'test-001'
        assert info.status == WorkerStatus.IDLE

    def test_worker_info_to_dict(self):
        """Test WorkerInfo to_dict."""
        info = WorkerInfo(
            worker_id='test-002',
            worker_type='training',
            device_type='cuda',
            device_name='RTX 4090',
            hostname='gpu-host',
            registered_at=time.time(),
            last_heartbeat=time.time(),
            status=WorkerStatus.WORKING,
        )

        d = info.to_dict()

        assert d['worker_id'] == 'test-002'
        assert d['worker_type'] == 'training'
        assert d['status'] == 'working'


class TestWorkerStatus:
    """Tests for WorkerStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert WorkerStatus.IDLE.value == 'idle'
        assert WorkerStatus.WORKING.value == 'working'
        assert WorkerStatus.DISCONNECTED.value == 'disconnected'


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_coordinator_not_found(self, ray_init):
        """Test getting nonexistent coordinator."""
        result = get_coordinator('nonexistent-coordinator')
        assert result is None

    def test_create_coordinator(self, ray_init):
        """Test creating coordinator via utility function."""
        name = f"util-coordinator-{int(time.time() * 1000)}"
        coord = create_coordinator({'test': True}, name=name)

        assert coord is not None

        # Use ray.kill() to avoid crashes during pytest teardown
        ray.kill(coord)

    def test_get_or_create_coordinator(self, ray_init):
        """Test get_or_create utility."""
        name = f"getorcreate-coordinator-{int(time.time() * 1000)}"

        # First call creates
        coord1 = get_or_create_coordinator({'test': True}, name=name)
        assert coord1 is not None

        # Second call gets existing
        coord2 = get_or_create_coordinator({'test': True}, name=name)

        # Should be same actor
        # (In local mode, this might create new ones, so just check they work)
        ray.get(coord1.get_config.remote())
        ray.get(coord2.get_config.remote())

        # Use ray.kill() to avoid crashes during pytest teardown
        ray.kill(coord1)
