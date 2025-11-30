#!/usr/bin/env python3
"""
Hybrid CPU/GPU Benchmark with batched inference.

Tests production-sized neural network with batch size 1 vs 128.
- 50 MCTS iterations per move
- 10 moves per game
- ResNet with 256 hidden, 10 blocks (production size)
"""

import os
import sys
import time
from functools import partial

# Set platform BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu,METAL'

import jax
import jax.numpy as jnp
import flax.linen as nn

# Get devices
_cpu_device = jax.devices('cpu')[0]
print(f"CPU Device: {_cpu_device}")

try:
    _gpu_device = jax.devices('METAL')[0]
    _has_gpu = True
    print(f"GPU Device: {_gpu_device}")
except Exception as e:
    _gpu_device = None
    _has_gpu = False
    print(f"No Metal GPU available: {e}")

import pgx.backgammon as bg
from core.types import StepMetadata
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector


# Pre-activation ResNet-V2 block (same as production)
class ResBlockV2(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        r = x
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features, use_bias=False)(x)
        return x + r


class ResNetTurboZero(nn.Module):
    """Production-sized ResNet."""
    num_actions: int = 156
    hidden_dim: int = 256
    num_blocks: int = 10

    @nn.compact
    def __call__(self, x):
        # ResNet tower
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_dim)(x)

        # Policy head
        policy_logits = nn.Dense(self.num_actions)(x)

        # Value head
        v = nn.LayerNorm()(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.squeeze(v, -1)

        return policy_logits, v


def create_hybrid_eval_fn(model, params, gpu_device, cpu_device):
    """Create eval function: GPU for NN, results back to CPU (batch_size=1)."""
    params_on_gpu = jax.device_put(params, gpu_device)

    @partial(jax.jit, device=gpu_device)
    def forward_on_gpu(obs):
        return model.apply(params_on_gpu, obs)

    def _gpu_inference(obs):
        policy_gpu, value_gpu = forward_on_gpu(obs)
        policy_cpu = jax.device_put(policy_gpu, cpu_device)
        value_cpu = jax.device_put(value_gpu, cpu_device)
        jax.block_until_ready(policy_cpu)
        jax.block_until_ready(value_cpu)
        return policy_cpu, value_cpu

    def eval_fn(env_state, params_arg, key):
        obs = env_state.observation
        policy, value = jax.pure_callback(
            _gpu_inference,
            (jax.ShapeDtypeStruct((model.num_actions,), jnp.float32),
             jax.ShapeDtypeStruct((), jnp.float32)),
            obs,
        )
        return policy, value

    return eval_fn


def create_hybrid_eval_fn_batched(model, params, gpu_device, cpu_device, batch_size):
    """Create batched eval function: accumulates batch_size states before GPU inference."""
    params_on_gpu = jax.device_put(params, gpu_device)

    @partial(jax.jit, device=gpu_device)
    def forward_on_gpu_batched(obs_batch):
        # obs_batch shape: (batch_size, obs_dim)
        return jax.vmap(lambda x: model.apply(params_on_gpu, x))(obs_batch)

    def _gpu_inference_batched(obs_batch):
        policy_gpu, value_gpu = forward_on_gpu_batched(obs_batch)
        policy_cpu = jax.device_put(policy_gpu, cpu_device)
        value_cpu = jax.device_put(value_gpu, cpu_device)
        jax.block_until_ready(policy_cpu)
        jax.block_until_ready(value_cpu)
        return policy_cpu, value_cpu

    # For MCTS, each call is still single-state, but we JIT the batched version
    # to measure the potential if we had batched MCTS
    def eval_fn(env_state, params_arg, key):
        obs = env_state.observation
        # Single inference through pure_callback
        policy, value = jax.pure_callback(
            lambda o: _gpu_inference_batched(o[None])[0][0],  # Add/remove batch dim
            (jax.ShapeDtypeStruct((model.num_actions,), jnp.float32),
             jax.ShapeDtypeStruct((), jnp.float32)),
            obs,
        )
        # Get first element from batch
        return policy, value

    return eval_fn, _gpu_inference_batched


def create_cpu_eval_fn(model, params):
    """Create CPU-only eval function."""
    @jax.jit
    def forward(params, obs):
        return model.apply(params, obs)

    def eval_fn(env_state, params_arg, key):
        obs = env_state.observation
        return forward(params_arg, obs)

    return eval_fn


def create_step_fn(env):
    """Create environment step function."""
    def step_fn(state, action, key):
        new_state = jax.lax.cond(
            state._is_stochastic,
            lambda: env.stochastic_step(state, action),
            lambda: env.step(state, action, key)
        )
        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count
        )
        return new_state, metadata

    return step_fn


def run_episode(env, mcts, params, max_steps=10, key=None):
    """Run a single episode using MCTS."""
    if key is None:
        key = jax.random.PRNGKey(42)

    state = env.init(key)
    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )
    eval_state = mcts.init(template_embedding=state)
    step_fn = create_step_fn(env)
    steps = 0

    for _ in range(max_steps):
        if state.terminated:
            break

        key, subkey = jax.random.split(key)
        output = mcts.evaluate(
            key=subkey,
            eval_state=eval_state,
            env_state=state,
            root_metadata=metadata,
            params=params,
            env_step_fn=step_fn,
        )

        action = output.action
        key, subkey = jax.random.split(key)
        state, metadata = step_fn(state, action, subkey)
        eval_state = mcts.step(output.eval_state, action)
        steps += 1

    return steps


def benchmark_mcts(label, env, mcts, params, warmup=1, episodes=3, max_steps=10):
    """Run MCTS benchmark."""
    print(f"\n{label}")
    print("-" * 60)

    key = jax.random.PRNGKey(0)

    # Warmup
    print("  JIT compiling (warmup)...", end=" ", flush=True)
    start = time.perf_counter()
    for i in range(warmup):
        key, subkey = jax.random.split(key)
        run_episode(env, mcts, params, max_steps=3, key=subkey)
    warmup_time = time.perf_counter() - start
    print(f"done ({warmup_time:.1f}s)")

    # Benchmark
    print("  Running benchmark...", end=" ", flush=True)
    total_steps = 0
    start = time.perf_counter()
    for i in range(episodes):
        key, subkey = jax.random.split(key)
        steps = run_episode(env, mcts, params, max_steps=max_steps, key=subkey)
        total_steps += steps
        print(f"{steps}", end=" ", flush=True)
    elapsed = time.perf_counter() - start
    print("done")

    steps_per_sec = total_steps / elapsed
    print(f"  Results: {total_steps} steps in {elapsed:.1f}s = {steps_per_sec:.3f} steps/sec")
    return {'steps_per_sec': steps_per_sec, 'total_steps': total_steps, 'elapsed': elapsed}


def benchmark_batched_inference(model, params, gpu_device, cpu_device, batch_sizes=[1, 128]):
    """Benchmark raw batched inference throughput (no MCTS)."""
    print("\n" + "=" * 60)
    print("Raw Batched Inference Benchmark (no MCTS overhead)")
    print("=" * 60)

    params_on_gpu = jax.device_put(params, gpu_device)
    sample_obs = jnp.zeros(34)

    results = {}
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")

        # Create batched forward function
        @partial(jax.jit, device=gpu_device)
        def forward_batched(obs_batch):
            return jax.vmap(lambda x: model.apply(params_on_gpu, x))(obs_batch)

        # Create batch
        obs_batch = jnp.stack([sample_obs] * batch_size)
        obs_batch = jax.device_put(obs_batch, cpu_device)

        # Warmup
        for _ in range(3):
            policy, value = forward_batched(obs_batch)
            jax.block_until_ready(policy)

        # Benchmark
        num_iters = 100
        start = time.perf_counter()
        for _ in range(num_iters):
            obs_batch_gpu = jax.device_put(obs_batch, gpu_device)
            policy, value = forward_batched(obs_batch_gpu)
            policy_cpu = jax.device_put(policy, cpu_device)
            jax.block_until_ready(policy_cpu)
        elapsed = time.perf_counter() - start

        inferences_per_sec = (num_iters * batch_size) / elapsed
        print(f"    {inferences_per_sec:.0f} inferences/sec ({num_iters * batch_size} total in {elapsed:.2f}s)")
        results[batch_size] = inferences_per_sec

    if len(batch_sizes) >= 2:
        speedup = results[batch_sizes[-1]] / results[batch_sizes[0]]
        print(f"\n  Batch {batch_sizes[-1]} vs {batch_sizes[0]}: {speedup:.1f}x throughput improvement")

    return results


def main():
    print("=" * 60)
    print("Hybrid CPU/GPU Benchmark - Production Network + Batching")
    print("=" * 60)

    # Configuration
    num_simulations = 50   # MCTS iterations per move
    max_nodes = 300        # Tree size
    max_steps = 10         # Moves per game
    num_episodes = 3       # Games to run

    print(f"\nConfiguration:")
    print(f"  Neural network: ResNet 256 hidden, 10 blocks")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Moves per game: {max_steps}")
    print(f"  Episodes: {num_episodes}")

    # Create environment
    env = bg.Backgammon(simple_doubles=True, short_game=True)

    # Create production-sized network
    model = ResNetTurboZero(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    sample_obs = jnp.zeros(34)
    params = model.init(key, sample_obs)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {param_count:,}")

    # First, benchmark raw batched inference (no MCTS)
    if _has_gpu:
        batch_results = benchmark_batched_inference(
            model, params, _gpu_device, _cpu_device, batch_sizes=[1, 128]
        )

    # ================================================================
    # MCTS Benchmarks
    # ================================================================
    print("\n" + "=" * 60)
    print("MCTS Benchmarks (batch_size=1 per MCTS iteration)")
    print("=" * 60)

    results = {}

    # CPU-only
    print("\nSetting up CPU-only MCTS...")
    with jax.default_device(_cpu_device):
        cpu_params = jax.device_put(params, _cpu_device)
        cpu_eval_fn = create_cpu_eval_fn(model, cpu_params)

        cpu_mcts = StochasticMCTS(
            eval_fn=cpu_eval_fn,
            action_selector=PUCTSelector(),
            stochastic_action_probs=env.stochastic_action_probs,
            branching_factor=env.num_actions,
            max_nodes=max_nodes,
            num_iterations=num_simulations,
        )

        results['cpu'] = benchmark_mcts(
            "CPU-only", env, cpu_mcts, cpu_params,
            warmup=1, episodes=num_episodes, max_steps=max_steps
        )

    # Hybrid
    if _has_gpu:
        print("\nSetting up Hybrid (CPU env + GPU NN) MCTS...")
        hybrid_eval_fn = create_hybrid_eval_fn(model, params, _gpu_device, _cpu_device)

        with jax.default_device(_cpu_device):
            hybrid_mcts = StochasticMCTS(
                eval_fn=hybrid_eval_fn,
                action_selector=PUCTSelector(),
                stochastic_action_probs=env.stochastic_action_probs,
                branching_factor=env.num_actions,
                max_nodes=max_nodes,
                num_iterations=num_simulations,
            )

            results['hybrid'] = benchmark_mcts(
                "Hybrid (CPU env + GPU NN)", env, hybrid_mcts, params,
                warmup=1, episodes=num_episodes, max_steps=max_steps
            )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nMCTS Performance (batch_size=1 per iteration):")
    print(f"  CPU-only: {results['cpu']['steps_per_sec']:.3f} steps/sec")
    if 'hybrid' in results:
        print(f"  Hybrid:   {results['hybrid']['steps_per_sec']:.3f} steps/sec")
        ratio = results['hybrid']['steps_per_sec'] / results['cpu']['steps_per_sec']
        if ratio > 1.1:
            print(f"\n  Hybrid is {ratio:.2f}x FASTER")
        elif ratio < 0.9:
            print(f"\n  CPU is {1/ratio:.2f}x FASTER")
        else:
            print("\n  Performance is similar")

    if _has_gpu and batch_results:
        print(f"\nRaw Inference Throughput:")
        print(f"  Batch=1:   {batch_results[1]:.0f} inferences/sec")
        print(f"  Batch=128: {batch_results[128]:.0f} inferences/sec")
        print(f"\n  Batching gives {batch_results[128]/batch_results[1]:.1f}x throughput")
        print("\n  Note: Current MCTS uses batch_size=1. Batched MCTS would benefit")
        print("  significantly from the increased inference throughput.")


if __name__ == '__main__':
    main()
