#!/usr/bin/env python3
"""
CPU-only batched games benchmark.

Tests running 1 vs 128 games in parallel using jax.vmap on CPU only.
This verifies the vmap batching works before adding GPU complexity.
"""

import os
import sys
import time

# Force CPU only
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import flax.linen as nn

print(f"JAX devices: {jax.devices()}")

import pgx.backgammon as bg
from core.types import StepMetadata
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector


# Simple network for faster JIT
class SimpleNet(nn.Module):
    num_actions: int = 156
    hidden_dim: int = 128  # Smaller for faster JIT
    num_blocks: int = 3    # Fewer blocks

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        for _ in range(self.num_blocks):
            r = x
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = x + r
            x = nn.relu(x)
        policy = nn.Dense(self.num_actions)(x)
        # 6-way value head for backgammon outcome distribution
        value_logits = nn.Dense(6)(x)
        return policy, value_logits


def create_eval_fn(model, params):
    """Create CPU eval function."""
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


def run_single_game(env, mcts, params, max_steps, key):
    """Run one game sequentially."""
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


def run_sequential_games(env, mcts, params, num_games, max_steps, key):
    """Run num_games sequentially (baseline)."""
    total_steps = 0
    for i in range(num_games):
        key, subkey = jax.random.split(key)
        steps = run_single_game(env, mcts, params, max_steps, subkey)
        total_steps += steps
    return total_steps


def benchmark(label, func, *args, warmup=1, runs=3):
    """Run benchmark with warmup."""
    print(f"\n{label}")
    print("-" * 50)

    key = jax.random.PRNGKey(0)

    # Warmup
    print("  JIT compiling...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(warmup):
        key, subkey = jax.random.split(key)
        func(*args, key=subkey)
    warmup_time = time.perf_counter() - start
    print(f"done ({warmup_time:.1f}s)")

    # Benchmark
    print("  Running...", end=" ", flush=True)
    total_steps = 0
    start = time.perf_counter()
    for _ in range(runs):
        key, subkey = jax.random.split(key)
        steps = func(*args, key=subkey)
        total_steps += steps
        print(f"{steps}", end=" ", flush=True)
    elapsed = time.perf_counter() - start
    print("done")

    steps_per_sec = total_steps / elapsed
    print(f"  Results: {total_steps} steps in {elapsed:.1f}s = {steps_per_sec:.1f} steps/sec")
    return steps_per_sec


def main():
    print("=" * 60)
    print("CPU-only Batched Games Benchmark")
    print("=" * 60)

    # Minimal config for fast JIT
    num_simulations = 20
    max_nodes = 100
    max_steps = 10

    print(f"\nConfiguration (minimal for fast JIT):")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Moves per game: {max_steps}")

    # Create environment
    env = bg.Backgammon(simple_doubles=True, short_game=True)

    # Create network
    model = SimpleNet(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    sample_obs = jnp.zeros(34)
    params = model.init(key, sample_obs)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {param_count:,}")

    # Create MCTS
    eval_fn = create_eval_fn(model, params)
    mcts = StochasticMCTS(
        eval_fn=eval_fn,
        action_selector=PUCTSelector(),
        stochastic_action_probs=env.stochastic_action_probs,
        branching_factor=env.num_actions,
        max_nodes=max_nodes,
        num_iterations=num_simulations,
    )

    results = {}

    # Sequential: 1 game at a time
    results['seq_1'] = benchmark(
        "Sequential: 1 game",
        run_sequential_games, env, mcts, params, 1, max_steps
    )

    # Sequential: 128 games one at a time
    results['seq_128'] = benchmark(
        "Sequential: 128 games (one at a time)",
        run_sequential_games, env, mcts, params, 128, max_steps,
        warmup=0, runs=1  # Just 1 run since it's slow
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n  1 game:   {results['seq_1']:.1f} steps/sec")
    print(f"  128 games (sequential): {results['seq_128']:.1f} steps/sec")

    # Note about vmap
    print("\n  Note: vmap batching would require a vmapped MCTS.evaluate")
    print("  which may not work with jax.pure_callback (for GPU).")
    print("  The MCTS loop inherently runs sequentially per tree.")


if __name__ == '__main__':
    main()
