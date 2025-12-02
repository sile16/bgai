"""Tests for the Coordinator and Redis state management.

No Ray dependency - uses Redis for all coordination.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from distributed.coordinator.head_node import Coordinator, create_coordinator
from distributed.coordinator.redis_state import (
    RedisStateManager,
    WorkerInfo,
    WorkerStatus,
    RunStatus,
    create_state_manager,
    WORKER_TTL,
)
from distributed.serialization import serialize_weights
import jax.numpy as jnp


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.smembers.return_value = set()
    mock.sadd.return_value = 1
    mock.srem.return_value = 1
    mock.delete.return_value = 1
    mock.exists.return_value = False
    mock.keys.return_value = []
    mock.llen.return_value = 0

    # Mock pipeline
    pipe_mock = MagicMock()
    pipe_mock.watch.return_value = None
    pipe_mock.get.return_value = b'0'
    pipe_mock.unwatch.return_value = None
    pipe_mock.multi.return_value = None
    pipe_mock.set.return_value = None
    pipe_mock.execute.return_value = [True, True]
    mock.pipeline.return_value = pipe_mock

    return mock


@pytest.fixture
def state_manager(mock_redis):
    """Create a RedisStateManager with mocked Redis."""
    with patch('distributed.coordinator.redis_state.redis.Redis', return_value=mock_redis):
        manager = RedisStateManager(host='localhost', port=6379)
        manager.redis = mock_redis
        yield manager


@pytest.fixture
def coordinator_config():
    """Sample coordinator configuration."""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'status_interval': 10.0,
        'mcts_simulations': 100,
        'mcts_max_nodes': 400,
        'train_batch_size': 128,
    }


@pytest.fixture
def sample_worker_info():
    """Sample WorkerInfo for testing."""
    return WorkerInfo(
        worker_id='test-worker-001',
        worker_type='game',
        device_type='cuda',
        device_name='RTX 4090',
        hostname='test-host',
        metrics_port=9100,
        status='idle',
        model_version=0,
    )


# ============================================================================
# WorkerInfo Tests
# ============================================================================

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
            metrics_port=9100,
        )

        assert info.worker_id == 'test-001'
        assert info.worker_type == 'game'
        assert info.status == 'idle'
        assert info.games_generated == 0
        assert info.model_version == 0

    def test_worker_info_to_dict(self):
        """Test WorkerInfo to_dict."""
        info = WorkerInfo(
            worker_id='test-002',
            worker_type='training',
            device_type='cuda',
            device_name='RTX 4090',
            hostname='gpu-host',
            metrics_port=9200,
            status='working',
        )

        d = info.to_dict()

        assert d['worker_id'] == 'test-002'
        assert d['worker_type'] == 'training'
        assert d['status'] == 'working'
        assert d['metrics_port'] == 9200

    def test_worker_info_to_json(self):
        """Test WorkerInfo JSON serialization."""
        info = WorkerInfo(
            worker_id='test-003',
            worker_type='eval',
            device_type='cpu',
            device_name='Intel Xeon',
            hostname='eval-host',
            metrics_port=9300,
        )

        json_str = info.to_json()
        restored = WorkerInfo.from_json(json_str)

        assert restored.worker_id == info.worker_id
        assert restored.worker_type == info.worker_type
        assert restored.device_type == info.device_type


class TestWorkerStatus:
    """Tests for WorkerStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert WorkerStatus.IDLE.value == 'idle'
        assert WorkerStatus.WORKING.value == 'working'
        assert WorkerStatus.DISCONNECTED.value == 'disconnected'


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_run_status_values(self):
        """Test run status enum values."""
        assert RunStatus.RUNNING.value == 'running'
        assert RunStatus.PAUSED.value == 'paused'
        assert RunStatus.STOPPED.value == 'stopped'


# ============================================================================
# RedisStateManager Tests
# ============================================================================

class TestRedisStateManager:
    """Tests for RedisStateManager."""

    def test_ping(self, state_manager, mock_redis):
        """Test Redis ping."""
        result = state_manager.ping()
        assert result is True
        mock_redis.ping.assert_called_once()

    def test_get_model_version_empty(self, state_manager, mock_redis):
        """Test getting model version when none set."""
        mock_redis.get.return_value = None
        version = state_manager.get_model_version()
        assert version == 0

    def test_get_model_version(self, state_manager, mock_redis):
        """Test getting model version."""
        mock_redis.get.return_value = b'5'
        version = state_manager.get_model_version()
        assert version == 5

    def test_set_model_version(self, state_manager, mock_redis):
        """Test setting model version."""
        state_manager.set_model_version(10)
        mock_redis.set.assert_called()

    def test_register_worker(self, state_manager, mock_redis, sample_worker_info):
        """Test worker registration."""
        state_manager.register_worker(sample_worker_info, ttl=60)

        # Verify setex was called (key with TTL)
        mock_redis.setex.assert_called()
        # Verify worker was added to set
        mock_redis.sadd.assert_called()

    def test_deregister_worker(self, state_manager, mock_redis):
        """Test worker deregistration."""
        result = state_manager.deregister_worker('test-worker')

        mock_redis.delete.assert_called()
        mock_redis.srem.assert_called()

    def test_heartbeat_worker_success(self, state_manager, mock_redis, sample_worker_info):
        """Test successful worker heartbeat."""
        # Setup mock to return existing worker data
        mock_redis.get.return_value = sample_worker_info.to_json().encode()

        result = state_manager.heartbeat_worker(
            'test-worker-001',
            stats={'status': 'working', 'games_generated': 10},
            ttl=60
        )

        assert result is True
        mock_redis.setex.assert_called()

    def test_heartbeat_worker_not_found(self, state_manager, mock_redis):
        """Test heartbeat for unknown worker."""
        mock_redis.get.return_value = None

        result = state_manager.heartbeat_worker('unknown-worker', {})

        assert result is False

    def test_get_run_status_default(self, state_manager, mock_redis):
        """Test getting run status when none set."""
        mock_redis.get.return_value = None
        status = state_manager.get_run_status()
        assert status == 'stopped'

    def test_set_run_status(self, state_manager, mock_redis):
        """Test setting run status."""
        state_manager.set_run_status(RunStatus.RUNNING)
        mock_redis.set.assert_called()

    def test_is_training_active(self, state_manager, mock_redis):
        """Test checking if training is active."""
        mock_redis.get.return_value = b'running'
        assert state_manager.is_training_active() is True

        mock_redis.get.return_value = b'stopped'
        assert state_manager.is_training_active() is False


# ============================================================================
# Warm Tree Tests
# ============================================================================

class TestWarmTree:
    """Tests for warm tree Redis state methods."""

    def test_get_warm_tree_empty(self, state_manager, mock_redis):
        """Test getting warm tree when none set."""
        mock_redis.get.return_value = None
        tree = state_manager.get_warm_tree()
        assert tree is None

    def test_get_warm_tree_version_empty(self, state_manager, mock_redis):
        """Test getting warm tree version when none set."""
        mock_redis.get.return_value = None
        version = state_manager.get_warm_tree_version()
        assert version == 0

    def test_set_warm_tree(self, state_manager, mock_redis):
        """Test setting warm tree."""
        tree_data = b'fake_tree_data'
        state_manager.set_warm_tree(tree_data, version=5)

        # Verify pipeline was used
        mock_redis.pipeline.assert_called()
        pipe = mock_redis.pipeline.return_value
        pipe.execute.assert_called()

    def test_get_warm_tree_if_newer(self, state_manager, mock_redis):
        """Test getting warm tree if newer version available."""
        mock_redis.get.side_effect = [b'10', b'fake_tree_data']  # version, tree

        result = state_manager.get_warm_tree_if_newer(current_version=5)

        assert result is not None
        tree_data, version = result
        assert version == 10
        assert tree_data == b'fake_tree_data'

    def test_get_warm_tree_if_newer_not_newer(self, state_manager, mock_redis):
        """Test get warm tree returns None if not newer."""
        mock_redis.get.return_value = b'5'  # version same as current

        result = state_manager.get_warm_tree_if_newer(current_version=5)

        assert result is None

    def test_delete_warm_tree(self, state_manager, mock_redis):
        """Test deleting warm tree."""
        state_manager.delete_warm_tree()

        # Should delete both keys
        assert mock_redis.delete.call_count >= 2


# ============================================================================
# Coordinator Tests (with mocked Redis)
# ============================================================================

class TestCoordinatorWithMockRedis:
    """Tests for Coordinator using mocked Redis."""

    def test_create_coordinator(self, mock_redis, coordinator_config):
        """Test creating coordinator."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            assert coord is not None
            assert coord.config == coordinator_config

    def test_coordinator_set_initial_weights(self, mock_redis, coordinator_config):
        """Test setting initial model weights."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.set_model_weights.return_value = True
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)

            weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})
            result = coord.set_initial_weights(weights)

            assert result['status'] == 'updated'
            assert result['version'] == 1

    def test_coordinator_update_model_weights(self, mock_redis, coordinator_config):
        """Test updating model weights."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.set_model_weights.return_value = True
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)

            weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})
            result = coord.update_model_weights(weights, version=5)

            assert result['status'] == 'updated'
            assert result['version'] == 5

    def test_coordinator_get_model_weights(self, mock_redis, coordinator_config):
        """Test getting model weights."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True

            weights = serialize_weights({'params': {'layer': jnp.ones((10, 10))}})
            mock_state.get_model_weights.return_value = weights
            mock_state.get_model_version.return_value = 3
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result_weights, version = coord.get_model_weights()

            assert version == 3
            assert result_weights is not None

    def test_coordinator_start_training_run(self, mock_redis, coordinator_config):
        """Test starting a training run."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_run_status.return_value = 'stopped'
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result = coord.start_training_run('test-run-001')

            assert result['status'] == 'started'
            assert result['run_id'] == 'test-run-001'

    def test_coordinator_start_training_run_already_running(self, mock_redis, coordinator_config):
        """Test starting a training run when one is already running."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_run_status.return_value = 'running'
            mock_state.get_run_id.return_value = 'existing-run'
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result = coord.start_training_run('new-run')

            assert result['status'] == 'error'
            assert 'already running' in result['reason']

    def test_coordinator_pause_training_run(self, mock_redis, coordinator_config):
        """Test pausing a training run."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_run_status.return_value = 'running'
            mock_state.get_run_id.return_value = 'test-run'
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result = coord.pause_training_run()

            assert result['status'] == 'paused'

    def test_coordinator_resume_training_run(self, mock_redis, coordinator_config):
        """Test resuming a training run."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_run_status.return_value = 'paused'
            mock_state.get_run_id.return_value = 'test-run'
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result = coord.resume_training_run()

            assert result['status'] == 'resumed'

    def test_coordinator_stop_training_run(self, mock_redis, coordinator_config):
        """Test stopping a training run."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_run_id.return_value = 'test-run'
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            result = coord.stop_training_run()

            assert result['status'] == 'stopped'

    def test_coordinator_get_cluster_status(self, mock_redis, coordinator_config):
        """Test getting cluster status."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_cluster_status.return_value = {
                'model_version': 5,
                'run_id': 'test-run',
                'run_status': 'running',
                'workers': {'game': 3, 'training': 1, 'eval': 1, 'total': 5},
                'active_workers': [],
                'total_games_generated': 1000,
                'total_training_steps': 500,
                'timestamp': time.time(),
            }
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            status = coord.get_cluster_status()

            assert 'model_version' in status
            assert 'workers' in status
            assert 'uptime_seconds' in status

    def test_coordinator_get_worker_counts(self, mock_redis, coordinator_config):
        """Test getting worker counts."""
        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_state.get_worker_counts.return_value = {
                'game': 3,
                'training': 1,
                'eval': 1,
                'total': 5,
            }
            mock_create.return_value = mock_state

            coord = Coordinator(coordinator_config)
            counts = coord.get_worker_counts()

            assert counts['game'] == 3
            assert counts['training'] == 1
            assert counts['total'] == 5


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_state_manager(self):
        """Test create_state_manager function."""
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
        }

        with patch('distributed.coordinator.redis_state.redis.Redis') as mock_redis:
            mock_redis.return_value.ping.return_value = True
            manager = create_state_manager(config)
            assert manager is not None

    def test_create_coordinator_function(self):
        """Test create_coordinator utility function."""
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
        }

        with patch('distributed.coordinator.head_node.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            coord = create_coordinator(config)
            assert coord is not None
            assert isinstance(coord, Coordinator)
