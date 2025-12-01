"""Metal-specific JAX tests for Apple Silicon (M1/M2/M3).

These tests verify that JAX with the Metal backend works correctly
for the operations needed in distributed AlphaZero training.
"""

import pytest
import jax
import jax.numpy as jnp
import time

from distributed.device import detect_device


# Skip all tests in this module if not on Metal
pytestmark = pytest.mark.skipif(
    not detect_device().is_metal,
    reason="Metal backend not available"
)


class TestMetalDetection:
    """Tests for Metal device detection."""

    def test_metal_detected(self):
        """Verify Metal backend is active."""
        info = detect_device()
        assert info.is_metal
        assert info.platform.upper() == 'METAL'

    def test_metal_is_gpu(self):
        """Verify Metal is recognized as GPU."""
        info = detect_device()
        assert info.is_gpu
        assert not info.is_cpu
        assert not info.is_cuda

    def test_metal_device_on_darwin(self):
        """Verify Metal only available on macOS."""
        info = detect_device()
        assert info.system_platform == 'Darwin'


class TestMetalBasicOps:
    """Tests for basic operations on Metal."""

    def test_metal_array_creation(self):
        """Test array creation on Metal."""
        x = jnp.zeros((1000, 1000))
        jax.block_until_ready(x)
        assert x.shape == (1000, 1000)

    def test_metal_matmul_small(self):
        """Test small matrix multiplication on Metal."""
        x = jnp.ones((100, 100))
        y = x @ x
        jax.block_until_ready(y)
        assert y.shape == (100, 100)
        assert jnp.allclose(y[0, 0], 100.0)

    def test_metal_matmul_medium(self):
        """Test medium matrix multiplication on Metal."""
        x = jnp.ones((500, 500))
        y = x @ x
        jax.block_until_ready(y)
        assert y.shape == (500, 500)

    @pytest.mark.slow
    def test_metal_matmul_large(self):
        """Test large matrix multiplication on Metal (stress test)."""
        x = jnp.ones((1000, 1000))
        y = x @ x
        jax.block_until_ready(y)
        assert y.shape == (1000, 1000)
        assert jnp.allclose(y[0, 0], 1000.0)

    def test_metal_elementwise_ops(self):
        """Test elementwise operations on Metal."""
        x = jnp.linspace(0, 10, 10000)

        # Chain of elementwise ops
        y = jnp.sin(x) + jnp.cos(x)
        y = jnp.exp(-y ** 2)
        y = jnp.tanh(y)

        jax.block_until_ready(y)
        assert y.shape == (10000,)


class TestMetalRandom:
    """Tests for random number generation on Metal."""

    def test_metal_random_normal(self):
        """Test normal distribution sampling."""
        key = jax.random.PRNGKey(42)
        samples = jax.random.normal(key, (10000,))
        jax.block_until_ready(samples)

        assert samples.shape == (10000,)
        # Check statistics are roughly correct
        assert abs(jnp.mean(samples)) < 0.05
        assert abs(jnp.std(samples) - 1.0) < 0.1

    def test_metal_random_uniform(self):
        """Test uniform distribution sampling."""
        key = jax.random.PRNGKey(42)
        samples = jax.random.uniform(key, (10000,), minval=0.0, maxval=1.0)
        jax.block_until_ready(samples)

        assert samples.shape == (10000,)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples <= 1.0)
        assert abs(jnp.mean(samples) - 0.5) < 0.05

    def test_metal_random_split(self):
        """Test random key splitting."""
        key = jax.random.PRNGKey(42)
        key1, key2 = jax.random.split(key)

        samples1 = jax.random.normal(key1, (100,))
        samples2 = jax.random.normal(key2, (100,))

        jax.block_until_ready(samples1)
        jax.block_until_ready(samples2)

        # Different keys should produce different samples
        assert not jnp.allclose(samples1, samples2)

    def test_metal_random_categorical(self):
        """Test categorical sampling (used for action selection)."""
        key = jax.random.PRNGKey(42)
        logits = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Sample many times
        keys = jax.random.split(key, 1000)
        samples = jax.vmap(lambda k: jax.random.categorical(k, logits))(keys)
        jax.block_until_ready(samples)

        # Higher logits should be sampled more often
        counts = jnp.bincount(samples, length=4)
        assert counts[3] > counts[0]  # logit 4.0 > logit 1.0


class TestMetalJIT:
    """Tests for JIT compilation on Metal."""

    def test_metal_jit_simple(self):
        """Test simple JIT function."""

        @jax.jit
        def add(x, y):
            return x + y

        x = jnp.ones((100,))
        y = jnp.ones((100,)) * 2

        result = add(x, y)
        jax.block_until_ready(result)

        assert jnp.allclose(result, jnp.ones((100,)) * 3)

    def test_metal_jit_with_matmul(self):
        """Test JIT with matrix multiplication."""

        @jax.jit
        def matmul_chain(x):
            y = x @ x
            z = y @ y
            return z

        x = jnp.eye(100)
        result = matmul_chain(x)
        jax.block_until_ready(result)

        # Identity matrix to any power is still identity
        assert jnp.allclose(result, jnp.eye(100))

    def test_metal_jit_recompilation(self):
        """Test JIT handles different input shapes."""

        @jax.jit
        def compute(x):
            return jnp.sum(x ** 2)

        # Different shapes trigger recompilation
        result1 = compute(jnp.ones((100,)))
        result2 = compute(jnp.ones((200,)))

        jax.block_until_ready(result1)
        jax.block_until_ready(result2)

        assert jnp.isclose(result1, 100.0)
        assert jnp.isclose(result2, 200.0)


class TestMetalVmap:
    """Tests for vmap (vectorization) on Metal."""

    def test_metal_vmap_simple(self):
        """Test simple vmap."""

        def single_fn(x):
            return jnp.sum(x ** 2)

        batched_fn = jax.vmap(single_fn)

        batch = jnp.ones((16, 100))
        result = batched_fn(batch)
        jax.block_until_ready(result)

        assert result.shape == (16,)
        assert jnp.allclose(result, jnp.ones(16) * 100)

    def test_metal_vmap_with_random(self):
        """Test vmap with random operations."""

        def sample_and_compute(key):
            samples = jax.random.normal(key, (100,))
            return jnp.mean(samples)

        keys = jax.random.split(jax.random.PRNGKey(42), 32)
        batched_fn = jax.vmap(sample_and_compute)

        result = batched_fn(keys)
        jax.block_until_ready(result)

        assert result.shape == (32,)

    def test_metal_nested_vmap(self):
        """Test nested vmap (batch of batches)."""

        def inner_fn(x):
            return jnp.sum(x)

        def outer_fn(batch):
            return jax.vmap(inner_fn)(batch)

        batched_outer = jax.vmap(outer_fn)

        # Shape: (4 outer batches, 8 inner batches, 10 elements)
        data = jnp.ones((4, 8, 10))
        result = batched_outer(data)
        jax.block_until_ready(result)

        assert result.shape == (4, 8)
        assert jnp.allclose(result, jnp.ones((4, 8)) * 10)


class TestMetalGrad:
    """Tests for automatic differentiation on Metal."""

    def test_metal_grad_simple(self):
        """Test simple gradient computation."""

        def loss(x):
            return jnp.sum(x ** 2)

        grad_fn = jax.grad(loss)
        x = jnp.array([1.0, 2.0, 3.0])
        grads = grad_fn(x)
        jax.block_until_ready(grads)

        assert jnp.allclose(grads, jnp.array([2.0, 4.0, 6.0]))

    def test_metal_grad_with_matmul(self):
        """Test gradient through matrix multiplication."""

        def loss(W, x):
            y = W @ x
            return jnp.sum(y ** 2)

        W = jnp.eye(10)
        x = jnp.ones(10)

        grad_fn = jax.grad(loss, argnums=0)
        grads = grad_fn(W, x)
        jax.block_until_ready(grads)

        assert grads.shape == (10, 10)

    def test_metal_value_and_grad(self):
        """Test value_and_grad on Metal."""

        def loss(params, x):
            W, b = params['W'], params['b']
            y = W @ x + b
            return jnp.sum(y ** 2)

        params = {
            'W': jnp.eye(10),
            'b': jnp.zeros(10),
        }
        x = jnp.ones(10)

        val_grad_fn = jax.value_and_grad(loss)
        value, grads = val_grad_fn(params, x)

        jax.block_until_ready(value)

        assert jnp.isclose(value, 10.0)  # sum((eye @ ones)^2) = sum(ones^2) = 10
        assert 'W' in grads
        assert 'b' in grads


class TestMetalLaxOps:
    """Tests for lax operations on Metal (used in MCTS/training)."""

    def test_metal_lax_cond(self):
        """Test lax.cond (used in stochastic game branching)."""

        def stochastic_branch(x):
            return x * 2

        def deterministic_branch(x):
            return x + 1

        # JIT compile the conditional
        @jax.jit
        def conditional_step(is_stochastic, x):
            return jax.lax.cond(
                is_stochastic,
                stochastic_branch,
                deterministic_branch,
                x
            )

        result_true = conditional_step(True, jnp.array(5.0))
        result_false = conditional_step(False, jnp.array(5.0))

        jax.block_until_ready(result_true)
        jax.block_until_ready(result_false)

        assert jnp.isclose(result_true, 10.0)
        assert jnp.isclose(result_false, 6.0)

    def test_metal_lax_scan(self):
        """Test lax.scan (used for sequential game steps)."""

        def step(carry, x):
            # Simulate game step: state update + action
            new_carry = carry + x
            output = new_carry * 2
            return new_carry, output

        @jax.jit
        def run_game(init_state, actions):
            final_state, outputs = jax.lax.scan(step, init_state, actions)
            return final_state, outputs

        init = jnp.array(0.0)
        actions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        final, outputs = run_game(init, actions)
        jax.block_until_ready(final)
        jax.block_until_ready(outputs)

        assert jnp.isclose(final, 15.0)  # 1+2+3+4+5
        assert outputs.shape == (5,)

    def test_metal_lax_fori_loop(self):
        """Test lax.fori_loop (used for MCTS iterations)."""

        @jax.jit
        def run_iterations(init_val, num_iters):
            def body_fn(i, val):
                return val + i

            return jax.lax.fori_loop(0, num_iters, body_fn, init_val)

        result = run_iterations(jnp.array(0.0), 100)
        jax.block_until_ready(result)

        # Sum of 0 to 99 = 99*100/2 = 4950
        assert jnp.isclose(result, 4950.0)


class TestMetalPerformance:
    """Performance tests on Metal (optional, marked slow)."""

    @pytest.mark.slow
    def test_metal_throughput_matmul(self):
        """Measure matrix multiplication throughput."""

        @jax.jit
        def batched_matmul(x):
            return x @ x.T

        x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))

        # Warmup
        _ = batched_matmul(x)
        jax.block_until_ready(_)

        # Timed run
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            result = batched_matmul(x)
        jax.block_until_ready(result)
        elapsed = time.time() - start

        ops_per_sec = num_iterations / elapsed
        print(f"\nMetal matmul throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  ({elapsed/num_iterations*1000:.2f} ms per 1000x1000 matmul)")

        # Should be able to do at least 10 ops/sec on M1
        assert ops_per_sec > 10

    @pytest.mark.slow
    def test_metal_throughput_vmap(self):
        """Measure vmap throughput."""

        @jax.jit
        def batched_forward(x):
            # Simulate neural network forward pass
            y = jnp.tanh(x @ jnp.ones((100, 256)))
            y = jnp.tanh(y @ jnp.ones((256, 256)))
            y = y @ jnp.ones((256, 1))
            return y

        batched_fn = jax.jit(jax.vmap(batched_forward))
        batch = jax.random.normal(jax.random.PRNGKey(0), (64, 100))

        # Warmup
        _ = batched_fn(batch)
        jax.block_until_ready(_)

        # Timed run
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            result = batched_fn(batch)
        jax.block_until_ready(result)
        elapsed = time.time() - start

        batches_per_sec = num_iterations / elapsed
        samples_per_sec = batches_per_sec * 64

        print(f"\nMetal vmap throughput: {batches_per_sec:.1f} batches/sec")
        print(f"  ({samples_per_sec:.0f} samples/sec for batch_size=64)")

        # Should be able to process at least 1000 samples/sec on M1
        assert samples_per_sec > 1000
