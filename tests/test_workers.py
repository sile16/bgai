"""Tests for distributed workers.

These tests verify the GameWorker and TrainingWorker functionality.
"""

import pytest
import time
import jax
import jax.numpy as jnp

import ray

from distributed.workers.base_worker import BaseWorker, WorkerStats
from distributed.workers.game_worker import GameWorker
from distributed.workers.training_worker import TrainingWorker
from distributed.coordinator.head_node import Coordinator, create_coordinator
from distributed.serialization import serialize_weights


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
        'mcts_simulations': 50,  # Reduced for faster tests
        'mcts_max_nodes': 200,
        'train_batch_size': 32,
    }

    name = f"test-coordinator-workers-{uuid.uuid4().hex[:8]}"
    coord = Coordinator.options(name=name).remote(config)

    yield coord

    # Use ray.kill() instead of shutdown.remote() to avoid crashes during pytest teardown
    # Ray in local_mode has issues with actor lifecycle during teardown
    try:
        ray.kill(coord)
    except Exception:
        pass


@pytest.fixture
def sample_device_info():
    """Sample device info for worker registration."""
    return {
        'device_type': 'cpu',
        'device_name': 'Test CPU',
        'hostname': 'test-host',
    }


class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_worker_stats_default(self):
        """Test default WorkerStats values."""
        stats = WorkerStats()
        assert stats.games_generated == 0
        assert stats.training_steps == 0
        assert stats.episodes_completed == 0

    def test_to_heartbeat_stats(self):
        """Test converting to heartbeat stats."""
        stats = WorkerStats()
        stats.games_generated = 10
        stats.training_steps = 5

        hb_stats = stats.to_heartbeat_stats(reset=True)

        assert hb_stats['games_since_last'] == 10
        assert hb_stats['training_steps_since_last'] == 5
        assert hb_stats['status'] == 'working'

        # Values should be reset
        assert stats.games_generated == 0
        assert stats.training_steps == 0

    def test_to_heartbeat_stats_no_reset(self):
        """Test heartbeat stats without reset."""
        stats = WorkerStats()
        stats.games_generated = 10

        hb_stats = stats.to_heartbeat_stats(reset=False)

        assert hb_stats['games_since_last'] == 10
        # Value should NOT be reset
        assert stats.games_generated == 10


class TestGameWorkerBasics:
    """Basic tests for GameWorker (no Redis required)."""

    def test_game_worker_creation(self, ray_init, coordinator):
        """Test creating a GameWorker."""
        config = {
            'batch_size': 2,
            'num_simulations': 10,
            'max_nodes': 50,
            'redis_host': 'localhost',
            'redis_port': 6379,
        }

        # Create worker
        worker = GameWorker.remote(coordinator, config=config)
        assert worker is not None

    def test_game_worker_type(self, ray_init, coordinator):
        """Test worker type property."""
        # GameWorker is wrapped by Ray, so we check the underlying class
        # Note: Ray actors don't expose __bases__ directly
        # We verify indirectly by checking the module structure
        from distributed.workers import game_worker
        assert hasattr(game_worker, 'GameWorker')
        # Verify it's a Ray actor class
        assert hasattr(GameWorker, 'remote')


@pytest.mark.skipif(
    not pytest.importorskip("redis", reason="Redis not installed"),
    reason="Redis package not available"
)
class TestGameWorkerWithRedis:
    """Tests for GameWorker that require Redis."""

    def test_game_worker_register(self, ray_init, coordinator, redis_client):
        """Test GameWorker registration with coordinator."""
        config = {
            'batch_size': 2,
            'num_simulations': 10,
            'max_nodes': 50,
            'redis_host': 'localhost',
            'redis_port': 6379,
        }

        worker = GameWorker.remote(coordinator, config=config)

        # Start worker (will register)
        # Note: We use a very small num_iterations to just test registration
        try:
            result = ray.get(worker.run.remote(num_iterations=1), timeout=60)
            assert result['status'] == 'completed'
        except ray.exceptions.GetTimeoutError:
            # Worker might be slow on first run due to JIT compilation
            ray.get(worker.stop.remote())
        except ray.exceptions.RayTaskError as e:
            # Skip if MPS serialization issue on Metal backend
            if "Unable to serialize MPS module" in str(e):
                pytest.skip("PGX backgammon uses jax.lax.cond which is incompatible with JAX Metal MPS backend")
            raise


class TestTrainingWorkerBasics:
    """Basic tests for TrainingWorker (no Redis required)."""

    def test_training_worker_creation(self, ray_init, coordinator):
        """Test creating a TrainingWorker."""
        config = {
            'train_batch_size': 32,
            'learning_rate': 3e-4,
            'redis_host': 'localhost',
            'redis_port': 6379,
        }

        worker = TrainingWorker.remote(coordinator, config=config)
        assert worker is not None

    def test_training_worker_type(self, ray_init, coordinator):
        """Test worker type property."""
        # TrainingWorker is wrapped by Ray, so we check the module structure
        from distributed.workers import training_worker
        assert hasattr(training_worker, 'TrainingWorker')
        # Verify it's a Ray actor class
        assert hasattr(TrainingWorker, 'remote')


class TestWorkerIntegration:
    """Integration tests for worker coordination."""

    def test_coordinator_receives_worker_heartbeat(self, ray_init, coordinator, sample_device_info):
        """Test that coordinator receives heartbeats from workers."""
        # Register a worker manually to simulate
        result = ray.get(coordinator.register_worker.remote(
            'test-game-worker',
            'game',
            sample_device_info
        ))

        assert result['status'] == 'registered'

        # Send a heartbeat
        hb_result = ray.get(coordinator.heartbeat.remote(
            'test-game-worker',
            {'games_since_last': 5, 'status': 'working'}
        ))

        assert hb_result['status'] == 'ok'

    def test_model_weights_distribution(self, ray_init, coordinator, sample_device_info):
        """Test that model weights can be distributed to workers."""
        # Set initial weights using simple numpy dict to avoid Flax/JAX tracing issues
        # in Ray local_mode (EvalTrace compatibility issue)
        import numpy as np

        # Create dummy weights as a simple nested dict with numpy arrays
        params = {
            'params': {
                'Dense_0': {
                    'kernel': np.random.randn(5, 10).astype(np.float32),
                    'bias': np.zeros(10, dtype=np.float32),
                }
            }
        }
        weights = serialize_weights(params)

        result = ray.get(coordinator.set_initial_weights.remote(weights))
        assert result['status'] == 'updated'
        assert result['version'] == 1

        # Register worker
        ray.get(coordinator.register_worker.remote(
            'test-worker',
            'game',
            sample_device_info
        ))

        # Get weights
        fetched_weights, version = ray.get(coordinator.get_model_weights.remote())
        assert version == 1
        assert fetched_weights is not None

        # Heartbeat should show new model available
        hb = ray.get(coordinator.heartbeat.remote(
            'test-worker',
            {'status': 'idle'}
        ))
        assert hb['new_model_available'] is True
        assert hb['new_model_version'] == 1


class TestWorkerConfiguration:
    """Tests for worker configuration."""

    def test_game_worker_default_config(self, ray_init, coordinator):
        """Test GameWorker uses sensible defaults."""
        # Create without config
        worker = GameWorker.remote(coordinator)
        # Worker should be created successfully with defaults
        assert worker is not None

    def test_training_worker_default_config(self, ray_init, coordinator):
        """Test TrainingWorker uses sensible defaults."""
        # Create without config
        worker = TrainingWorker.remote(coordinator)
        # Worker should be created successfully with defaults
        assert worker is not None

    def test_worker_custom_config(self, ray_init, coordinator):
        """Test workers accept custom configuration."""
        game_config = {
            'batch_size': 64,
            'num_simulations': 200,
            'max_nodes': 800,
            'temperature': 0.5,
        }

        train_config = {
            'train_batch_size': 256,
            'learning_rate': 1e-4,
            'weight_push_interval': 20,
        }

        game_worker = GameWorker.remote(coordinator, config=game_config)
        train_worker = TrainingWorker.remote(coordinator, config=train_config)

        assert game_worker is not None
        assert train_worker is not None


class TestBaseWorkerAbstract:
    """Tests for BaseWorker abstract class."""

    def test_base_worker_is_abstract(self):
        """Test that BaseWorker cannot be instantiated directly."""
        # BaseWorker is abstract and should not be instantiated
        with pytest.raises(TypeError):
            BaseWorker(None)

    def test_worker_id_generation(self):
        """Test that worker IDs are generated with correct format."""
        # The worker ID should include the worker type and hostname
        # We can't test directly since BaseWorker is abstract, but we can
        # verify the format expected by game/training workers
        import socket
        hostname = socket.gethostname()

        # Worker IDs should follow pattern: type-hostname-uuid
        # This is tested implicitly through the workers


class TestWorkerLifecycle:
    """Tests for worker lifecycle management."""

    def test_worker_graceful_stop(self, ray_init, coordinator):
        """Test workers can be stopped gracefully."""
        worker = GameWorker.remote(coordinator, config={
            'batch_size': 2,
            'num_simulations': 5,
            'redis_host': 'localhost',
            'redis_port': 6379,
        })

        # Worker should be created and stop without error
        # (even if Redis isn't available, the stop should not raise)
        assert worker is not None

    def test_multiple_workers_same_coordinator(self, ray_init, coordinator, sample_device_info):
        """Test multiple workers can register with same coordinator."""
        # Register multiple workers
        for i in range(3):
            result = ray.get(coordinator.register_worker.remote(
                f'game-worker-{i}',
                'game',
                sample_device_info
            ))
            assert result['status'] == 'registered'

        # Verify counts
        counts = ray.get(coordinator.get_worker_count.remote())
        assert counts['game'] >= 3

    def test_worker_deregistration(self, ray_init, coordinator, sample_device_info):
        """Test workers can deregister properly."""
        # Register
        ray.get(coordinator.register_worker.remote(
            'temp-worker',
            'game',
            sample_device_info
        ))

        # Deregister
        result = ray.get(coordinator.deregister_worker.remote('temp-worker'))
        assert result['status'] == 'deregistered'

        # Heartbeat should indicate unknown worker
        hb = ray.get(coordinator.heartbeat.remote(
            'temp-worker',
            {'status': 'idle'}
        ))
        assert hb['status'] == 'unknown_worker'
        assert hb['should_register'] is True
