#!/usr/bin/env python3
"""
Benchmark comparing CPU-only vs Hybrid (CPU env + GPU NN) for StochasticMCTS.

This benchmark measures:
1. Full CPU: Everything runs on CPU
2. Hybrid: env.step on CPU, neural network forward pass on GPU (Metal)

Usage:
    python benchmarks/bench_hybrid_mcts.py
"""

import os
import sys
import time
from functools import partial
from typing import Tuple, Optional, Callable

# Set platform BEFORE importing JAX - need both CPU and Metal
os.environ['JAX_PLATFORMS'] = 'cpu,METAL'

import jax
import jax.numpy as jnp
import chex
import flax.linen as nn

# Get devices
_cpu_device = jax.devices('cpu')[0]
print(f"CPU Device: {_cpu_device}")

# Try to get Metal device
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


class SimpleResNet(nn.Module):
    """Simple ResNet for backgammon evaluation."""
    num_actions: int = 156
    num_hidden: int = 256
    num_blocks: int = 6

    @nn.compact
    def __call__(self, x):
        # Input layer
        h = nn.Dense(self.num_hidden)(x)
        h = nn.relu(h)

        # Residual blocks
        for _ in range(self.num_blocks):
            residual = h
            h = nn.Dense(self.num_hidden)(h)
            h = nn.relu(h)
            h = nn.Dense(self.num_hidden)(h)
            h = nn.relu(h + residual)

        # Policy head
        policy = nn.Dense(self.num_actions)(h)

        # 6-way value head for backgammon outcome distribution
        value_logits = nn.Dense(6)(h)

        return policy, value_logits


def create_nn_eval_fn(model, params, state_to_input_fn, gpu_device=None, cpu_device=None):
    """Create evaluation function that runs NN on GPU but returns results on CPU.

    The eval_fn signature must be: (env_state, params, key) -> (policy_logits, value)
    This matches what turbozero's AlphaZero evaluator expects.

    For hybrid mode:
    - NN params live on GPU
    - Input is transferred to GPU for inference
    - Results are transferred back to CPU so MCTS operations stay on CPU
    - Uses jax.pure_callback to isolate GPU ops from JAX tracing
      (this avoids the lax.cond Metal bug in turbozero)
    """

    if gpu_device is not None and cpu_device is not None:
        # Place params on GPU
        params_on_gpu = jax.device_put(params, gpu_device)

        # Create a JIT-compiled forward pass that runs on GPU
        @partial(jax.jit, device=gpu_device)
        def forward_on_gpu(obs):
            return model.apply(params_on_gpu, obs)

        def _gpu_inference(obs):
            """Run inference on GPU and return CPU arrays."""
            # Run on GPU
            policy_gpu, value_gpu = forward_on_gpu(obs)
            # Transfer back to CPU and block
            policy_cpu = jax.device_put(policy_gpu, cpu_device)
            value_cpu = jax.device_put(value_gpu, cpu_device)
            jax.block_until_ready(policy_cpu)
            jax.block_until_ready(value_cpu)
            return policy_cpu, value_cpu

        def eval_fn(env_state, params_arg, key):
            # Convert state to NN input
            obs = state_to_input_fn(env_state)

            # Use pure_callback to isolate GPU ops from JAX tracing
            # This makes the GPU inference opaque to JAX's compiler
            policy, value = jax.pure_callback(
                _gpu_inference,
                (jax.ShapeDtypeStruct((model.num_actions,), jnp.float32),
                 jax.ShapeDtypeStruct((), jnp.float32)),
                obs,
            )
            return policy, value
    else:
        # CPU-only mode
        @jax.jit
        def forward(params, obs):
            return model.apply(params, obs)

        def eval_fn(env_state, params_arg, key):
            obs = state_to_input_fn(env_state)
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


def run_mcts_episode(env, mcts, params, max_steps=100, key=None):
    """Run a single episode using MCTS, return steps completed."""
    if key is None:
        key = jax.random.PRNGKey(42)

    state = env.init(key)

    # Get initial metadata
    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )

    # Initialize MCTS tree
    eval_state = mcts.init(template_embedding=state)

    step_fn = create_step_fn(env)
    steps = 0

    for _ in range(max_steps):
        if state.terminated:
            break

        key, subkey = jax.random.split(key)

        # Run MCTS evaluation
        output = mcts.evaluate(
            key=subkey,
            eval_state=eval_state,
            env_state=state,
            root_metadata=metadata,
            params=params,
            env_step_fn=step_fn,
        )

        action = output.action

        # Take action in environment
        key, subkey = jax.random.split(key)
        state, metadata = step_fn(state, action, subkey)

        # Advance MCTS tree to the subtree corresponding to the action taken
        eval_state = mcts.step(output.eval_state, action)
        steps += 1

    return steps


def benchmark_config(env, mcts, params, num_episodes=5, warmup=2):
    """Benchmark a specific MCTS configuration."""
    key = jax.random.PRNGKey(0)

    # Warmup
    print("  Warming up...", end=" ", flush=True)
    for i in range(warmup):
        key, subkey = jax.random.split(key)
        run_mcts_episode(env, mcts, params, max_steps=20, key=subkey)
    print("done")

    # Benchmark
    total_steps = 0
    start = time.perf_counter()

    for i in range(num_episodes):
        key, subkey = jax.random.split(key)
        steps = run_mcts_episode(env, mcts, params, max_steps=50, key=subkey)
        total_steps += steps
        print(f"  Episode {i+1}/{num_episodes}: {steps} steps")

    elapsed = time.perf_counter() - start

    return {
        'total_steps': total_steps,
        'elapsed': elapsed,
        'steps_per_sec': total_steps / elapsed,
        'episodes': num_episodes,
    }


def main():
    print("=" * 70)
    print("Hybrid CPU/GPU Benchmark for StochasticMCTS")
    print("=" * 70)
    print()

    # Configuration - reduced for faster testing
    num_simulations = 20  # MCTS simulations per move
    max_nodes = 100
    num_episodes = 3

    print(f"Configuration:")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Episodes: {num_episodes}")
    print()

    # Create environment (disable short_game for longer episodes)
    env = bg.Backgammon(simple_doubles=True, short_game=False)

    # Create neural network
    model = SimpleResNet(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    sample_obs = jnp.zeros(34)  # Backgammon observation size
    params = model.init(key, sample_obs)

    state_to_input_fn = lambda state: state.observation

    # ================================================================
    # Benchmark 1: Full CPU
    # ================================================================
    print("-" * 70)
    print("Benchmark 1: Full CPU")
    print("-" * 70)

    with jax.default_device(_cpu_device):
        cpu_params = jax.device_put(params, _cpu_device)
        cpu_eval_fn = create_nn_eval_fn(model, cpu_params, state_to_input_fn)

        cpu_mcts = StochasticMCTS(
            eval_fn=cpu_eval_fn,
            action_selector=PUCTSelector(),
            stochastic_action_probs=env.stochastic_action_probs,
            branching_factor=env.num_actions,
            max_nodes=max_nodes,
            num_iterations=num_simulations,
        )

        cpu_results = benchmark_config(env, cpu_mcts, cpu_params, num_episodes=num_episodes)

    print(f"\n  Results: {cpu_results['steps_per_sec']:.1f} steps/sec")
    print()

    # ================================================================
    # Benchmark 2: Hybrid (CPU env + GPU NN) - if GPU available
    # ================================================================
    if _has_gpu:
        print("-" * 70)
        print("Benchmark 2: Hybrid (CPU env + GPU neural network)")
        print("-" * 70)

        # Create hybrid eval function: GPU for NN inference, results back to CPU
        gpu_eval_fn = create_nn_eval_fn(model, params, state_to_input_fn,
                                        gpu_device=_gpu_device, cpu_device=_cpu_device)

        # MCTS runs on CPU but uses GPU eval_fn
        with jax.default_device(_cpu_device):
            hybrid_mcts = StochasticMCTS(
                eval_fn=gpu_eval_fn,
                action_selector=PUCTSelector(),
                stochastic_action_probs=env.stochastic_action_probs,
                branching_factor=env.num_actions,
                max_nodes=max_nodes,
                num_iterations=num_simulations,
            )

            hybrid_results = benchmark_config(env, hybrid_mcts, params, num_episodes=num_episodes)

        print(f"\n  Results: {hybrid_results['steps_per_sec']:.1f} steps/sec")
        print()

        # ================================================================
        # Summary
        # ================================================================
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"  Full CPU:     {cpu_results['steps_per_sec']:>8.1f} steps/sec")
        print(f"  Hybrid:       {hybrid_results['steps_per_sec']:>8.1f} steps/sec")

        speedup = hybrid_results['steps_per_sec'] / cpu_results['steps_per_sec']
        if speedup > 1:
            print(f"\n  Hybrid is {speedup:.2f}x FASTER than CPU")
        else:
            print(f"\n  CPU is {1/speedup:.2f}x FASTER than Hybrid")

        print()
        print("Recommendation:")
        if speedup > 1.1:
            print("  -> Use HYBRID mode on Mac (GPU for NN, CPU for env)")
        elif speedup < 0.9:
            print("  -> Use CPU-only mode on Mac")
        else:
            print("  -> Performance is similar, either mode works")
    else:
        print("No GPU available - skipping hybrid benchmark")
        print(f"\nCPU-only result: {cpu_results['steps_per_sec']:.1f} steps/sec")


if __name__ == '__main__':
    main()
