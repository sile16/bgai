#!/usr/bin/env python3
"""
Simple Hybrid CPU/GPU Benchmark - minimal MCTS to verify approach works.

This benchmark uses very small MCTS parameters to minimize JIT compilation time
while still verifying the hybrid CPU+GPU approach works correctly.
"""

import os
import sys
import time

# Set platform BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu,METAL'

import jax
import jax.numpy as jnp
from functools import partial

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

import flax.linen as nn
import pgx.backgammon as bg
from core.types import StepMetadata
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector


class TinyNet(nn.Module):
    """Minimal network for fast JIT."""
    num_actions: int = 156

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(64)(x)
        h = nn.relu(h)
        policy = nn.Dense(self.num_actions)(h)
        # 6-way value head for backgammon outcome distribution
        value_logits = nn.Dense(6)(h)
        return policy, value_logits


def create_hybrid_eval_fn(model, params, gpu_device, cpu_device):
    """Create eval function: GPU for NN, results back to CPU."""
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


def run_episode(env, mcts, params, max_steps=30, key=None):
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


def benchmark(label, env, mcts, params, warmup=1, episodes=2, max_steps=20):
    """Run benchmark with given configuration."""
    print(f"\n{label}")
    print("-" * 50)

    key = jax.random.PRNGKey(0)

    # Warmup (includes JIT compilation)
    print("  JIT compiling (warmup)...", end=" ", flush=True)
    start = time.perf_counter()
    for i in range(warmup):
        key, subkey = jax.random.split(key)
        run_episode(env, mcts, params, max_steps=5, key=subkey)
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
    elapsed = time.perf_counter() - start
    print("done")

    steps_per_sec = total_steps / elapsed
    print(f"  Results: {total_steps} steps in {elapsed:.1f}s = {steps_per_sec:.2f} steps/sec")
    return steps_per_sec


def main():
    print("=" * 60)
    print("Simple Hybrid CPU/GPU Benchmark")
    print("=" * 60)

    # Very minimal config to reduce JIT time
    num_simulations = 5  # Minimal MCTS simulations
    max_nodes = 30       # Minimal tree size

    print(f"\nConfiguration (minimal for fast JIT):")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")

    # Create environment
    env = bg.Backgammon(simple_doubles=True, short_game=True)

    # Create tiny network
    model = TinyNet(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    sample_obs = jnp.zeros(34)
    params = model.init(key, sample_obs)

    # Test CPU-only
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

        cpu_result = benchmark("CPU-only", env, cpu_mcts, cpu_params)

    # Test Hybrid (if GPU available)
    if _has_gpu:
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

            hybrid_result = benchmark("Hybrid (CPU env + GPU NN)", env, hybrid_mcts, params)

        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  CPU-only: {cpu_result:.2f} steps/sec")
        print(f"  Hybrid:   {hybrid_result:.2f} steps/sec")

        if hybrid_result > cpu_result * 1.1:
            print(f"\n  Hybrid is {hybrid_result/cpu_result:.2f}x FASTER")
        elif cpu_result > hybrid_result * 1.1:
            print(f"\n  CPU is {cpu_result/hybrid_result:.2f}x FASTER")
        else:
            print("\n  Performance is similar")
    else:
        print("\nNo GPU available - CPU-only result shown")


if __name__ == '__main__':
    main()
