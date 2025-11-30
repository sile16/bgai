"""Pytest configuration and shared fixtures for distributed training tests."""

import pytest
import jax
import jax.numpy as jnp
from typing import Generator, Optional

from distributed.device import detect_device, DeviceInfo


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device_info() -> DeviceInfo:
    """Provide device information for the current platform."""
    return detect_device()


@pytest.fixture(scope="session")
def is_metal(device_info: DeviceInfo) -> bool:
    """Check if running on Metal (Apple Silicon)."""
    return device_info.is_metal


@pytest.fixture(scope="session")
def is_cuda(device_info: DeviceInfo) -> bool:
    """Check if running on CUDA (NVIDIA GPU)."""
    return device_info.is_cuda


@pytest.fixture(scope="session")
def is_cpu(device_info: DeviceInfo) -> bool:
    """Check if running on CPU only."""
    return device_info.is_cpu


# ============================================================================
# JAX Fixtures
# ============================================================================

@pytest.fixture
def rng_key() -> jax.Array:
    """Provide a fresh random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_observation() -> jax.Array:
    """Create sample backgammon observation (shape matches PGX)."""
    # PGX backgammon observation has shape (34,) - flattened board representation
    return jnp.zeros((34,), dtype=jnp.float32)


@pytest.fixture
def sample_batch_observation() -> jax.Array:
    """Create batched sample observations."""
    batch_size = 16
    return jnp.zeros((batch_size, 34), dtype=jnp.float32)


@pytest.fixture
def sample_experience() -> dict:
    """Create sample experience for testing replay buffer."""
    return {
        'observation_nn': jnp.zeros((34,), dtype=jnp.float32),
        'policy_weights': jnp.ones((156,), dtype=jnp.float32) / 156,
        'policy_mask': jnp.ones((156,), dtype=jnp.bool_),
        'cur_player_id': jnp.int32(0),
        'reward': jnp.array([1.0, -1.0], dtype=jnp.float32),
    }


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def redis_available() -> bool:
    """Check if Redis is available."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379)
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def redis_client(redis_available: bool):
    """Provide Redis client, skip if unavailable."""
    if not redis_available:
        pytest.skip("Redis not available")

    import redis
    client = redis.Redis(host='localhost', port=6379, decode_responses=False)

    yield client

    # Cleanup test keys after all tests
    for key in client.keys("test:*"):
        client.delete(key)


# ============================================================================
# Ray Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing (local mode for isolation)."""
    import ray

    if not ray.is_initialized():
        ray.init(local_mode=True, ignore_reinit_error=True)

    yield ray

    # Don't shutdown - let other test modules reuse


# ============================================================================
# Skip Markers
# ============================================================================

# Markers for conditional test execution
metal_only = pytest.mark.skipif(
    not detect_device().is_metal,
    reason="Test requires Metal backend"
)

cuda_only = pytest.mark.skipif(
    not detect_device().is_cuda,
    reason="Test requires CUDA backend"
)

gpu_only = pytest.mark.skipif(
    not detect_device().is_gpu,
    reason="Test requires GPU (Metal or CUDA)"
)

requires_redis = pytest.mark.skipif(
    True,  # Will be overridden by fixture
    reason="Test requires Redis server"
)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "metal_only: mark test to run only on Metal backend"
    )
    config.addinivalue_line(
        "markers", "cuda_only: mark test to run only on CUDA backend"
    )
    config.addinivalue_line(
        "markers", "gpu_only: mark test to run only on GPU (Metal or CUDA)"
    )
    config.addinivalue_line(
        "markers", "requires_redis: mark test to run only when Redis is available"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
