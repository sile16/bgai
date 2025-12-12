#!/usr/bin/env python3
"""
Benchmark eval worker performance: GPU vs CPU at different batch sizes.

This script measures the time taken to run neural network inference
(the core operation in eval) for both GPU and CPU at various batch sizes.

Usage:
    python scripts/benchmark_eval_gpu_vs_cpu.py
"""

import os
import sys
import time
import subprocess
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 60)
    print("  Eval Worker Benchmark: GPU vs CPU")
    print("=" * 60)
    print()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    num_iterations = 100

    # Common benchmark code
    benchmark_code = '''
import time
import json
import jax
import jax.numpy as jnp
from core.networks.mlp import MLP, MLPConfig

jax.devices()
print(f"JAX device: {{jax.devices()[0]}}", file=__import__('sys').stderr)

# Create network matching distributed config
# hidden_dim=256, num_blocks=6 translates to hidden_dims=[256]*6
config = MLPConfig(
    hidden_dims=[256] * 6,
    policy_head_out_size=156,
    value_head_out_size=6,  # default: win/lose/gammon/backgammon probs
)
network = MLP(config)

# Initialize network with dummy input to get params (Flax linen pattern)
dummy_input = jnp.zeros((1, 198), dtype=jnp.float32)
rng = jax.random.PRNGKey(0)
params = network.init(rng, dummy_input, train=False)

# Warmup JIT with apply
_ = network.apply(params, dummy_input, train=False)
jax.block_until_ready(_)

results = {{}}
batch_sizes = {batch_sizes}
num_iterations = {num_iterations}

for batch_size in batch_sizes:
    print(f"  Batch size {{batch_size}}...", file=__import__('sys').stderr, end=" ", flush=True)
    batch_input = jnp.zeros((batch_size, 198), dtype=jnp.float32)

    # Warmup for this batch size
    for _ in range(5):
        policy, value = network.apply(params, batch_input, train=False)
        jax.block_until_ready(policy)
        jax.block_until_ready(value)

    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        policy, value = network.apply(params, batch_input, train=False)
        jax.block_until_ready(policy)
        jax.block_until_ready(value)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    throughput = batch_size / avg_time

    results[batch_size] = {{
        "avg_ms": avg_time * 1000,
        "min_ms": min_time * 1000,
        "throughput": throughput,
    }}
    print(f"avg={{avg_time*1000:.2f}}ms, throughput={{throughput:.0f}} samples/s", file=__import__('sys').stderr)

print(json.dumps(results))
'''

    # Run CPU benchmark
    print("Running CPU benchmark...")
    cpu_env = os.environ.copy()
    cpu_env["JAX_PLATFORMS"] = "cpu"
    cpu_env["PYTHONPATH"] = PROJECT_ROOT + ":" + cpu_env.get("PYTHONPATH", "")

    cpu_script = benchmark_code.format(batch_sizes=batch_sizes, num_iterations=num_iterations)

    result = subprocess.run(
        [sys.executable, "-c", cpu_script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=cpu_env,
    )
    print(result.stderr)

    if result.returncode != 0:
        print(f"CPU benchmark failed: {result.stderr}")
        cpu_results = {}
    else:
        try:
            cpu_results = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Failed to parse CPU results: {result.stdout}")
            cpu_results = {}

    print()

    # Run GPU benchmark
    print("Running GPU benchmark...")
    gpu_env = os.environ.copy()
    gpu_env.pop("JAX_PLATFORMS", None)
    gpu_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
    gpu_env["PYTHONPATH"] = PROJECT_ROOT + ":" + gpu_env.get("PYTHONPATH", "")

    gpu_script = benchmark_code.format(batch_sizes=batch_sizes, num_iterations=num_iterations)

    result = subprocess.run(
        [sys.executable, "-c", gpu_script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=gpu_env,
    )
    print(result.stderr)

    if result.returncode != 0:
        print(f"GPU benchmark failed: {result.stderr}")
        gpu_results = {}
    else:
        try:
            gpu_results = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Failed to parse GPU results: {result.stdout}")
            gpu_results = {}

    # Print comparison table
    print()
    print("=" * 80)
    print("  RESULTS: GPU vs CPU Comparison")
    print("=" * 80)
    print()
    print(f"{'Batch':>6} | {'CPU avg':>10} | {'GPU avg':>10} | {'Speedup':>8} | {'CPU tput':>12} | {'GPU tput':>12}")
    print("-" * 80)

    for batch_size in batch_sizes:
        cpu_key = str(batch_size)
        gpu_key = str(batch_size)

        if cpu_key in cpu_results and gpu_key in gpu_results:
            cpu_avg = cpu_results[cpu_key]["avg_ms"]
            gpu_avg = gpu_results[gpu_key]["avg_ms"]
            cpu_tput = cpu_results[cpu_key]["throughput"]
            gpu_tput = gpu_results[gpu_key]["throughput"]
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

            print(f"{batch_size:>6} | {cpu_avg:>8.2f}ms | {gpu_avg:>8.2f}ms | {speedup:>7.2f}x | {cpu_tput:>10.0f}/s | {gpu_tput:>10.0f}/s")
        else:
            print(f"{batch_size:>6} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {'N/A':>12} | {'N/A':>12}")

    print()
    print("Recommendation:")
    if gpu_results and cpu_results:
        # Find the batch size with best GPU speedup
        best_speedup = 0
        best_batch = 8
        for batch_size in batch_sizes:
            cpu_key = str(batch_size)
            gpu_key = str(batch_size)
            if cpu_key in cpu_results and gpu_key in gpu_results:
                cpu_avg = cpu_results[cpu_key]["avg_ms"]
                gpu_avg = gpu_results[gpu_key]["avg_ms"]
                speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_batch = batch_size

        if best_speedup > 1.5:
            print(f"  GPU is {best_speedup:.1f}x faster at batch_size={best_batch}. Use GPU for eval.")
        else:
            print(f"  GPU speedup is minimal ({best_speedup:.1f}x). CPU may be sufficient for eval.")


if __name__ == "__main__":
    main()
