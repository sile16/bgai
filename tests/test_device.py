"""Tests for device detection and JAX functionality."""

import pytest
import jax
import jax.numpy as jnp

from distributed.device import (
    detect_device,
    get_device_config,
    DeviceInfo,
    DeviceConfig,
    CUDA_CONFIG,
    METAL_CONFIG,
    CPU_CONFIG,
)


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_detect_device_returns_device_info(self):
        """Verify detect_device returns DeviceInfo instance."""
        info = detect_device()
        assert isinstance(info, DeviceInfo)

    def test_detect_device_has_valid_platform(self):
        """Verify platform is one of the expected values."""
        info = detect_device()
        # Platform can be 'cpu', 'gpu', 'METAL', etc.
        assert info.platform is not None
        assert len(info.platform) > 0

    def test_detect_device_has_device_count(self):
        """Verify at least one device is detected."""
        info = detect_device()
        assert info.device_count >= 1

    def test_detect_device_flags_are_mutually_consistent(self):
        """Verify device flags are consistent."""
        info = detect_device()

        # Can't be both CUDA and Metal
        assert not (info.is_cuda and info.is_metal)

        # If GPU, must be either CUDA or Metal
        if info.is_gpu:
            assert info.is_cuda or info.is_metal

        # If CPU only, not GPU
        if info.is_cpu:
            assert not info.is_gpu

    def test_detect_device_has_jax_version(self):
        """Verify JAX version is captured."""
        info = detect_device()
        assert info.jax_version is not None
        assert len(info.jax_version) > 0

    def test_detect_device_has_system_platform(self):
        """Verify system platform is captured."""
        info = detect_device()
        assert info.system_platform in ('Darwin', 'Linux', 'Windows')

    def test_device_info_str_representation(self):
        """Verify DeviceInfo has useful string representation."""
        info = detect_device()
        info_str = str(info)
        assert len(info_str) > 0
        # Should contain one of the device types
        assert any(x in info_str for x in ['CUDA', 'Metal', 'CPU'])


class TestDeviceConfig:
    """Tests for device configuration."""

    def test_get_device_config_returns_config(self):
        """Verify get_device_config returns DeviceConfig."""
        config = get_device_config()
        assert isinstance(config, DeviceConfig)

    def test_get_device_config_has_positive_values(self):
        """Verify all config values are positive."""
        config = get_device_config()
        assert config.mcts_simulations > 0
        assert config.mcts_max_nodes > 0
        assert config.game_batch_size > 0
        assert config.train_batch_size > 0

    def test_get_device_config_matches_device(self, device_info):
        """Verify config matches detected device type."""
        config = get_device_config(device_info)

        if device_info.is_cuda:
            assert config == CUDA_CONFIG
        elif device_info.is_metal:
            assert config == METAL_CONFIG
        else:
            assert config == CPU_CONFIG

    def test_config_to_dict(self):
        """Verify config can be converted to dictionary."""
        config = get_device_config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'mcts_simulations' in config_dict
        assert 'mcts_max_nodes' in config_dict
        assert 'game_batch_size' in config_dict
        assert 'train_batch_size' in config_dict

    def test_cuda_config_is_most_aggressive(self):
        """Verify CUDA config has highest batch sizes."""
        assert CUDA_CONFIG.train_batch_size >= METAL_CONFIG.train_batch_size
        assert CUDA_CONFIG.train_batch_size >= CPU_CONFIG.train_batch_size
        assert CUDA_CONFIG.game_batch_size >= METAL_CONFIG.game_batch_size
        assert CUDA_CONFIG.game_batch_size >= CPU_CONFIG.game_batch_size

    def test_cpu_config_is_most_conservative(self):
        """Verify CPU config has lowest batch sizes."""
        assert CPU_CONFIG.train_batch_size <= METAL_CONFIG.train_batch_size
        assert CPU_CONFIG.train_batch_size <= CUDA_CONFIG.train_batch_size


class TestJaxBasicOperations:
    """Tests for basic JAX operations on the current device."""

    def test_jax_array_creation(self):
        """Verify basic JAX array creation."""
        x = jnp.ones((100, 100))
        assert x.shape == (100, 100)
        assert x.dtype == jnp.float32

    def test_jax_matmul(self):
        """Verify matrix multiplication works."""
        x = jnp.ones((100, 100))
        y = jnp.dot(x, x)
        jax.block_until_ready(y)
        assert y.shape == (100, 100)
        # Each element should be 100 (sum of 100 ones)
        assert jnp.allclose(y[0, 0], 100.0)

    def test_jax_random(self, rng_key):
        """Verify random number generation."""
        samples = jax.random.normal(rng_key, (1000,))
        jax.block_until_ready(samples)
        assert samples.shape == (1000,)
        # Mean should be approximately 0
        assert abs(jnp.mean(samples)) < 0.2

    def test_jax_jit_compilation(self):
        """Verify JIT compilation works."""

        @jax.jit
        def matmul(x):
            return x @ x

        x = jnp.ones((100, 100))

        # First call compiles
        result = matmul(x)
        jax.block_until_ready(result)

        # Second call uses compiled version
        result2 = matmul(x)
        jax.block_until_ready(result2)

        assert result.shape == (100, 100)
        assert jnp.allclose(result, result2)

    def test_jax_vmap(self):
        """Verify vmap works (critical for batched game playing)."""

        def single_fn(x):
            return x * 2 + 1

        batched_fn = jax.vmap(single_fn)

        batch = jnp.ones((16, 100))
        result = batched_fn(batch)
        jax.block_until_ready(result)

        assert result.shape == (16, 100)
        # Each element should be 2*1 + 1 = 3
        assert jnp.allclose(result[0, 0], 3.0)

    def test_jax_grad(self):
        """Verify automatic differentiation works."""

        def loss_fn(x):
            return jnp.sum(x ** 2)

        grad_fn = jax.grad(loss_fn)

        x = jnp.array([1.0, 2.0, 3.0])
        grads = grad_fn(x)
        jax.block_until_ready(grads)

        # Gradient of sum(x^2) is 2*x
        assert grads.shape == (3,)
        assert jnp.allclose(grads, jnp.array([2.0, 4.0, 6.0]))

    def test_jax_value_and_grad(self):
        """Verify value_and_grad works (used in training)."""

        def loss_fn(x):
            return jnp.sum(x ** 2)

        value_and_grad_fn = jax.value_and_grad(loss_fn)

        x = jnp.array([1.0, 2.0, 3.0])
        value, grads = value_and_grad_fn(x)
        jax.block_until_ready(value)
        jax.block_until_ready(grads)

        assert jnp.isclose(value, 14.0)  # 1 + 4 + 9
        assert jnp.allclose(grads, jnp.array([2.0, 4.0, 6.0]))

    def test_jax_lax_cond(self):
        """Verify lax.cond works (used in stochastic game logic)."""

        def true_fn(x):
            return x * 2

        def false_fn(x):
            return x + 1

        # Test true branch
        result_true = jax.lax.cond(True, true_fn, false_fn, jnp.array(5.0))
        assert jnp.isclose(result_true, 10.0)

        # Test false branch
        result_false = jax.lax.cond(False, true_fn, false_fn, jnp.array(5.0))
        assert jnp.isclose(result_false, 6.0)

    def test_jax_scan(self):
        """Verify lax.scan works (used in sequential operations)."""

        def step_fn(carry, x):
            new_carry = carry + x
            output = new_carry
            return new_carry, output

        init_carry = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])

        final_carry, outputs = jax.lax.scan(step_fn, init_carry, xs)
        jax.block_until_ready(final_carry)
        jax.block_until_ready(outputs)

        assert jnp.isclose(final_carry, 10.0)  # 1+2+3+4
        assert outputs.shape == (4,)
        assert jnp.allclose(outputs, jnp.array([1.0, 3.0, 6.0, 10.0]))

    def test_jax_tree_map(self):
        """Verify tree_map works (used for param manipulation)."""
        params = {
            'layer1': {'w': jnp.ones((10, 10)), 'b': jnp.zeros((10,))},
            'layer2': {'w': jnp.ones((10, 5)), 'b': jnp.zeros((5,))},
        }

        # Scale all parameters by 2
        scaled_params = jax.tree_util.tree_map(lambda x: x * 2, params)

        assert jnp.allclose(scaled_params['layer1']['w'], jnp.ones((10, 10)) * 2)
        assert jnp.allclose(scaled_params['layer2']['b'], jnp.zeros((5,)))


class TestJaxDeviceAwareness:
    """Tests for device-aware JAX operations."""

    def test_array_on_correct_device(self, device_info):
        """Verify arrays are created on the detected device."""
        x = jnp.ones((100, 100))
        # JAX arrays should be on the default device
        # Note: .device is a property in JAX 0.4.x, method in later versions
        device = x.device() if callable(x.device) else x.device
        assert device is not None

    def test_jit_compilation_on_device(self, device_info):
        """Verify JIT compilation targets the correct device."""

        @jax.jit
        def compute(x):
            return jnp.sin(x) + jnp.cos(x)

        x = jnp.linspace(0, 10, 1000)
        result = compute(x)
        jax.block_until_ready(result)

        assert result.shape == (1000,)

    @pytest.mark.slow
    def test_larger_computation(self, device_info):
        """Test larger computation to verify device utilization."""

        @jax.jit
        def heavy_compute(x):
            for _ in range(10):
                x = jnp.tanh(x @ x.T)
            return x

        x = jax.random.normal(jax.random.PRNGKey(0), (500, 500))
        result = heavy_compute(x)
        jax.block_until_ready(result)

        assert result.shape == (500, 500)
