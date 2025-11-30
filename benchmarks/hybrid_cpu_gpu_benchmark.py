#!/usr/bin/env python
"""Benchmark hybrid CPU/GPU vs full GPU for AlphaZero-style workloads.

This script helps determine whether splitting env.step() to CPU and NN to GPU
is faster than running everything on GPU.

Usage:
    # Full GPU mode
    JAX_PLATFORMS=cuda python benchmarks/hybrid_cpu_gpu_benchmark.py

    # Or for Metal
    JAX_PLATFORMS=METAL,cpu python benchmarks/hybrid_cpu_gpu_benchmark.py

Results will show whether hybrid or full-GPU is faster for your hardware.
"""

import jax
import jax.numpy as jnp
import time
import argparse


def get_devices():
    """Get CPU and GPU devices."""
    cpu = jax.devices('cpu')[0]

    # Try CUDA first, then Metal
    try:
        gpu = jax.devices('cuda')[0]
        gpu_type = 'CUDA'
    except RuntimeError:
        try:
            gpu = jax.devices('METAL')[0]
            gpu_type = 'Metal'
        except RuntimeError:
            gpu = None
            gpu_type = None

    return cpu, gpu, gpu_type


def create_nn_params(key, num_hidden=256, num_blocks=6, num_actions=156):
    """Create AlphaZero-style NN parameters."""
    keys = jax.random.split(key, 2 + num_blocks * 2)

    params = {
        'input_w': jax.random.normal(keys[0], (34, num_hidden)) * 0.1,
        'input_b': jnp.zeros(num_hidden),
        'blocks': [],
        'policy_w': jax.random.normal(keys[1], (num_hidden, num_actions)) * 0.1,
        'value_w': jax.random.normal(keys[1], (num_hidden, 1)) * 0.1,
    }

    for i in range(num_blocks):
        params['blocks'].append({
            'w1': jax.random.normal(keys[2 + i*2], (num_hidden, num_hidden)) * 0.1,
            'w2': jax.random.normal(keys[3 + i*2], (num_hidden, num_hidden)) * 0.1,
        })

    return params


def nn_forward(params, x):
    """Forward pass through AlphaZero-style network."""
    h = x @ params['input_w'] + params['input_b']
    h = jax.nn.relu(h)

    for block in params['blocks']:
        residual = h
        h = jax.nn.relu(h @ block['w1'])
        h = h @ block['w2']
        h = jax.nn.relu(h + residual)

    policy = h @ params['policy_w']
    value = h @ params['value_w']
    return policy, value


def env_step_sim(state):
    """Simulate env.step() - small element-wise operations."""
    # Flip board simulation
    new_state = state.at[:, :24].set(jnp.flip(state[:, :24], axis=1))
    new_state = new_state * -1

    # Legal action mask computation (simplified)
    mask = new_state[:, :26] > 0

    # State updates
    new_state = new_state.at[:, 27].add(1)  # step count

    return new_state, mask


def benchmark_full_gpu(gpu, batch_size, num_iters=200):
    """Benchmark everything on GPU."""
    key = jax.random.PRNGKey(42)

    with jax.default_device(gpu):
        state = jax.random.normal(key, (batch_size, 28))
        obs = jax.random.normal(key, (batch_size, 34))
        params = create_nn_params(key)

    env_step_jit = jax.jit(env_step_sim)
    nn_forward_jit = jax.jit(nn_forward)

    # Warmup
    for _ in range(20):
        new_state, mask = env_step_jit(state)
        policy, value = nn_forward_jit(params, obs)
        jax.block_until_ready(value)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        new_state, mask = env_step_jit(state)
        jax.block_until_ready(new_state)
        policy, value = nn_forward_jit(params, obs)
        jax.block_until_ready(value)
    elapsed = time.perf_counter() - start

    return elapsed / num_iters


def benchmark_hybrid(cpu, gpu, batch_size, num_iters=200):
    """Benchmark hybrid: env on CPU, NN on GPU."""
    key = jax.random.PRNGKey(42)

    # State on CPU, NN params on GPU
    with jax.default_device(cpu):
        state = jax.random.normal(key, (batch_size, 28))
        obs_cpu = jax.random.normal(key, (batch_size, 34))

    with jax.default_device(gpu):
        params = create_nn_params(key)

    env_step_jit = jax.jit(env_step_sim)
    nn_forward_jit = jax.jit(nn_forward)

    # Warmup
    for _ in range(20):
        with jax.default_device(cpu):
            new_state, mask = env_step_jit(state)
        obs_gpu = jax.device_put(obs_cpu, gpu)
        policy, value = nn_forward_jit(params, obs_gpu)
        jax.block_until_ready(value)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        # CPU: env step
        with jax.default_device(cpu):
            new_state, mask = env_step_jit(state)
        jax.block_until_ready(new_state)

        # Transfer to GPU
        obs_gpu = jax.device_put(obs_cpu, gpu)

        # GPU: NN forward
        policy, value = nn_forward_jit(params, obs_gpu)
        jax.block_until_ready(value)
    elapsed = time.perf_counter() - start

    return elapsed / num_iters


def benchmark_full_cpu(cpu, batch_size, num_iters=200):
    """Benchmark everything on CPU."""
    key = jax.random.PRNGKey(42)

    with jax.default_device(cpu):
        state = jax.random.normal(key, (batch_size, 28))
        obs = jax.random.normal(key, (batch_size, 34))
        params = create_nn_params(key)

    env_step_jit = jax.jit(env_step_sim)
    nn_forward_jit = jax.jit(nn_forward)

    # Warmup
    for _ in range(20):
        new_state, mask = env_step_jit(state)
        policy, value = nn_forward_jit(params, obs)
        jax.block_until_ready(value)

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        new_state, mask = env_step_jit(state)
        jax.block_until_ready(new_state)
        policy, value = nn_forward_jit(params, obs)
        jax.block_until_ready(value)
    elapsed = time.perf_counter() - start

    return elapsed / num_iters


def main():
    parser = argparse.ArgumentParser(description='Benchmark hybrid vs full GPU')
    parser.add_argument('--batch-sizes', type=str, default='1,8,32,128,512,1024,2048',
                        help='Comma-separated batch sizes to test')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations per benchmark')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    cpu, gpu, gpu_type = get_devices()

    print(f"CPU Device: {cpu}")
    print(f"GPU Device: {gpu} ({gpu_type})")
    print(f"Default backend: {jax.default_backend()}")
    print()

    if gpu is None:
        print("No GPU found, running CPU-only benchmark")
        print()
        print("Full CPU Results:")
        print("-" * 50)
        for bs in batch_sizes:
            time_s = benchmark_full_cpu(cpu, bs, args.num_iters)
            throughput = bs / time_s
            print(f"Batch {bs:5d}: {time_s*1000:>7.2f} ms  ({throughput:>12,.0f} samples/sec)")
        return

    # Run all benchmarks
    print("=" * 70)
    print(f"{'Batch':<8} {'Full GPU':<18} {'Hybrid':<18} {'Full CPU':<18} {'Winner'}")
    print("=" * 70)

    for bs in batch_sizes:
        gpu_time = benchmark_full_gpu(gpu, bs, args.num_iters)
        hybrid_time = benchmark_hybrid(cpu, gpu, bs, args.num_iters)
        cpu_time = benchmark_full_cpu(cpu, bs, args.num_iters)

        gpu_throughput = bs / gpu_time
        hybrid_throughput = bs / hybrid_time
        cpu_throughput = bs / cpu_time

        # Determine winner
        times = {'GPU': gpu_time, 'Hybrid': hybrid_time, 'CPU': cpu_time}
        winner = min(times, key=times.get)
        speedup = max(gpu_time, hybrid_time, cpu_time) / min(gpu_time, hybrid_time, cpu_time)

        print(f"{bs:<8} {gpu_throughput:>10,.0f}/s       {hybrid_throughput:>10,.0f}/s       {cpu_throughput:>10,.0f}/s       {winner} ({speedup:.1f}x)")

    print("=" * 70)
    print()
    print("Recommendations:")
    print("-" * 70)
    print("• If 'Hybrid' wins at your typical batch size, use CPU for env.step()")
    print("• If 'GPU' wins, keep everything on GPU (fix the lax.cond bug in PGX)")
    print("• RTX 4090 typically needs batch >= 256 for GPU to beat hybrid")
    print("• For MCTS with batched evaluation, larger batches = more GPU benefit")


if __name__ == '__main__':
    main()
