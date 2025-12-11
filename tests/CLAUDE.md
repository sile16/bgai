# tests/ - Test Suite

Pytest-based tests for distributed training system.

## Running Tests

```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_redis_buffer.py

# With verbose output
pytest tests/ -v
```

## Test Files

- **test_redis_buffer.py**: Replay buffer operations, FIFO eviction, sampling
- **test_serialization.py**: Experience/weight serialization roundtrips
- **test_workers.py**: Worker lifecycle, heartbeat, registration
- **test_coordinator.py**: Coordinator state management
- **test_config_loader.py**: YAML config parsing, device overrides
- **test_device.py**: Device detection (CUDA, Metal, CPU)
- **test_jax_metal.py**: JAX Metal backend compatibility (skip if no Metal)

## Fixtures (`conftest.py`)

- `redis_client`: Fresh Redis connection (flushes test keys)
- `config`: Default test configuration
- `temp_checkpoint_dir`: Temporary directory for checkpoints

## Requirements

- Redis server running locally (or set `REDIS_HOST`)
- `pytest` and `pytest-asyncio` installed

## Writing Tests

```python
def test_buffer_add_experience(redis_client, config):
    buffer = RedisReplayBuffer(**config['redis'])
    buffer.add(experience)
    assert buffer.size() == 1
```

Tests should:
- Use fixtures for shared resources
- Clean up Redis keys after test
- Mock external services when possible
