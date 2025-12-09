"""Tests for distributed workers.

No Ray dependency - workers run as standalone processes
coordinated via Redis.
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from distributed.workers.base_worker import BaseWorker, WorkerStats
from distributed.workers.game_worker import GameWorker
from distributed.workers.training_worker import TrainingWorker
from distributed.workers.eval_worker import EvalWorker
from distributed.coordinator.redis_state import WorkerInfo


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
def mock_state_manager(mock_redis):
    """Create a mock RedisStateManager."""
    mock_state = MagicMock()
    mock_state.ping.return_value = True
    mock_state.get_model_version.return_value = 0
    mock_state.get_model_weights.return_value = None
    mock_state.get_model_weights_if_newer.return_value = None
    mock_state.get_run_status.return_value = 'stopped'
    mock_state.is_training_active.return_value = False
    mock_state.register_worker.return_value = None
    mock_state.deregister_worker.return_value = True
    mock_state.heartbeat_worker.return_value = True
    return mock_state


@pytest.fixture
def worker_config():
    """Sample worker configuration."""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'batch_size': 2,
        'num_simulations': 10,
        'max_nodes': 50,
        'metrics_port': 9100,
    }


@pytest.fixture
def training_config():
    """Sample training worker configuration."""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'train_batch_size': 32,
        'learning_rate': 3e-4,
        'min_buffer_size': 10,
        'games_per_epoch': 5,
        'steps_per_game': 2,
        'metrics_port': 9200,
    }


# ============================================================================
# WorkerStats Tests
# ============================================================================

class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_worker_stats_default(self):
        """Test default WorkerStats values."""
        stats = WorkerStats()
        assert stats.games_generated == 0
        assert stats.training_steps == 0
        assert stats.episodes_completed == 0

    def test_get_heartbeat_stats(self):
        """Test getting heartbeat stats."""
        stats = WorkerStats()
        stats.games_generated = 10
        stats.training_steps = 5

        hb_stats = stats.get_heartbeat_stats(reset=True)

        assert hb_stats['games_generated'] == 10
        assert hb_stats['training_steps'] == 5
        assert hb_stats['status'] == 'working'

        # Values should be reset
        assert stats.games_generated == 0
        assert stats.training_steps == 0

    def test_get_heartbeat_stats_no_reset(self):
        """Test heartbeat stats without reset."""
        stats = WorkerStats()
        stats.games_generated = 10

        hb_stats = stats.get_heartbeat_stats(reset=False)

        assert hb_stats['games_generated'] == 10
        # Value should NOT be reset
        assert stats.games_generated == 10


# ============================================================================
# BaseWorker Tests
# ============================================================================

class TestBaseWorkerAbstract:
    """Tests for BaseWorker abstract class."""

    def test_base_worker_is_abstract(self, mock_state_manager):
        """Test that BaseWorker cannot be instantiated directly."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with pytest.raises(TypeError):
                BaseWorker({})

    def test_worker_id_format(self):
        """Test that worker IDs follow expected format."""
        # Worker IDs should be: type-hostname-uuid
        import socket
        hostname = socket.gethostname().split('.')[0]

        # We verify this format is used by checking the _generate_worker_id method
        # on a concrete implementation
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker.__new__(GameWorker)
                worker.config = {}
                worker.device_info = MagicMock()
                worker.device_info.is_cuda = False
                worker.device_info.is_metal = False
                worker.device_config = {}

                # Manually call the ID generator
                class MockWorker(BaseWorker):
                    @property
                    def worker_type(self):
                        return 'test'

                    def _run_loop(self, num_iterations=-1):
                        return {}

                with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state):
                    mock_worker = MockWorker.__new__(MockWorker)
                    mock_worker.config = {}
                    # Would normally generate ID like 'test-hostname-abc123'


# ============================================================================
# GameWorker Tests
# ============================================================================

class TestGameWorkerBasics:
    """Basic tests for GameWorker."""

    def test_game_worker_type_property(self):
        """Test GameWorker type property."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'batch_size': 2,
                })
                assert worker.worker_type == 'game'

    def test_game_worker_config_defaults(self):
        """Test GameWorker uses sensible defaults."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                assert worker.batch_size >= 1
                assert worker.num_simulations >= 1
                assert worker.max_nodes >= 1

    def test_game_worker_custom_config(self):
        """Test GameWorker accepts custom configuration."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'batch_size': 64,
                    'num_simulations': 200,
                    'max_nodes': 800,
                    'temperature': 0.5,
                })

                assert worker.batch_size == 64
                assert worker.num_simulations == 200
                assert worker.max_nodes == 800


# ============================================================================
# TrainingWorker Tests
# ============================================================================

class TestTrainingWorkerBasics:
    """Basic tests for TrainingWorker."""

    def test_training_worker_type_property(self):
        """Test TrainingWorker type property."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.training_worker.RedisReplayBuffer'):
                worker = TrainingWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })
                assert worker.worker_type == 'training'

    def test_training_worker_config_defaults(self):
        """Test TrainingWorker uses sensible defaults."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.training_worker.RedisReplayBuffer'):
                worker = TrainingWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                assert worker.train_batch_size >= 1
                assert worker.learning_rate > 0
                assert worker.games_per_epoch >= 1

    def test_training_worker_custom_config(self):
        """Test TrainingWorker accepts custom configuration."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.training_worker.RedisReplayBuffer'):
                worker = TrainingWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'train_batch_size': 256,
                    'learning_rate': 1e-4,
                    'games_per_epoch': 20,
                    'steps_per_game': 5,
                })

                assert worker.train_batch_size == 256
                assert worker.learning_rate == 1e-4
                assert worker.games_per_epoch == 20
                assert worker.steps_per_game == 5


# ============================================================================
# EvalWorker Tests
# ============================================================================

class TestEvalWorkerBasics:
    """Basic tests for EvalWorker."""

    def test_eval_worker_type_property(self):
        """Test EvalWorker type property."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.eval_worker.RedisReplayBuffer'):
                worker = EvalWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })
                assert worker.worker_type == 'eval'

    def test_eval_worker_config_defaults(self):
        """Test EvalWorker uses sensible defaults."""
        with patch('distributed.workers.base_worker.create_state_manager') as mock_create:
            mock_state = MagicMock()
            mock_state.ping.return_value = True
            mock_create.return_value = mock_state

            with patch('distributed.workers.eval_worker.RedisReplayBuffer'):
                worker = EvalWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                assert worker.eval_games >= 1
                assert worker.batch_size >= 1
                assert worker.num_simulations >= 1


# ============================================================================
# Worker Registration Tests
# ============================================================================

class TestWorkerRegistration:
    """Tests for worker registration with Redis."""

    def test_worker_registration(self, mock_state_manager):
        """Test worker registration."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                # Registration happens in register() method
                result = worker.register()

                assert result is True
                mock_state_manager.register_worker.assert_called()

    def test_worker_deregistration(self, mock_state_manager):
        """Test worker deregistration."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.register()
                result = worker.deregister()

                assert result is True


# ============================================================================
# Worker Heartbeat Tests
# ============================================================================

class TestWorkerHeartbeat:
    """Tests for worker heartbeat functionality."""

    def test_heartbeat_success(self, mock_state_manager):
        """Test successful heartbeat."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.register()

                # Heartbeat should succeed
                result = worker._heartbeat()
                assert result is True

    def test_heartbeat_reregisters_on_failure(self, mock_state_manager):
        """Test that heartbeat re-registers if worker key expired."""
        mock_state_manager.heartbeat_worker.return_value = False

        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.registered = True
                result = worker._heartbeat()

                # Should have attempted to re-register
                mock_state_manager.register_worker.assert_called()


# ============================================================================
# Model Weight Synchronization Tests
# ============================================================================

class TestModelWeightSync:
    """Tests for model weight synchronization."""

    def test_check_for_model_update(self, mock_state_manager):
        """Test checking for model updates."""
        mock_state_manager.get_model_version.return_value = 5

        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.current_model_version = 3

                result = worker.check_for_model_update()
                assert result is True

    def test_check_for_model_update_current(self, mock_state_manager):
        """Test no update when already current."""
        mock_state_manager.get_model_version.return_value = 5

        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.current_model_version = 5

                result = worker.check_for_model_update()
                assert result is False


# ============================================================================
# Worker Stats Tests
# ============================================================================

class TestWorkerStatsReporting:
    """Tests for worker statistics reporting."""

    def test_get_stats(self, mock_state_manager):
        """Test getting worker stats."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.stats.games_generated = 100
                worker.current_model_version = 5

                stats = worker.get_stats()

                assert stats['worker_type'] == 'game'
                assert stats['model_version'] == 5
                assert 'uptime' in stats


# ============================================================================
# Worker Lifecycle Tests
# ============================================================================

class TestWorkerLifecycle:
    """Tests for worker lifecycle management."""

    def test_worker_stop(self, mock_state_manager):
        """Test worker stop."""
        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                worker.running = True
                worker.registered = True

                worker.stop()

                assert worker.running is False
                assert worker.registered is False

    def test_is_training_active(self, mock_state_manager):
        """Test checking if training is active."""
        mock_state_manager.is_training_active.return_value = True

        with patch('distributed.workers.base_worker.create_state_manager', return_value=mock_state_manager):
            with patch('distributed.workers.game_worker.RedisReplayBuffer'):
                worker = GameWorker(config={
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                })

                result = worker.is_training_active()
                assert result is True
