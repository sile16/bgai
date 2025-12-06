#!/usr/bin/env python3
"""
Properly vmapped games benchmark following turbozero's pattern.

Uses jax.vmap over the MCTS evaluate to run multiple games in parallel.
"""

import os
import sys
import time
from functools import partial

# Force CPU only for now
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import flax.linen as nn

print(f"JAX devices: {jax.devices()}")

import pgx.backgammon as bg
from core.types import StepMetadata
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector


# Simple network
class SimpleNet(nn.Module):
    num_actions: int = 156
    hidden_dim: int = 64
    num_blocks: int = 2

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


def make_step_fn(env):
    """Create step function (not jitted - will be vmapped)."""
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


def single_game_step(mcts, step_fn, params, state, metadata, eval_state, key):
    """Take one step in a single game."""
    key, eval_key, step_key = jax.random.split(key, 3)

    # Run MCTS
    output = mcts.evaluate(
        key=eval_key,
        eval_state=eval_state,
        env_state=state,
        root_metadata=metadata,
        params=params,
        env_step_fn=step_fn,
    )

    # Take action
    action = output.action
    new_state, new_metadata = step_fn(state, action, step_key)
    new_eval_state = mcts.step(output.eval_state, action)

    return new_state, new_metadata, new_eval_state, key


def run_vmapped_games(env, mcts, params, batch_size, num_steps, key):
    """Run batch_size games for num_steps using vmap."""
    # Initialize batch of games
    keys = jax.random.split(key, batch_size + 1)
    key, init_keys = keys[0], keys[1:]

    # Vectorized init
    states = jax.vmap(env.init)(init_keys)

    # Create batched metadata
    metadatas = jax.vmap(lambda s: StepMetadata(
        rewards=s.rewards,
        action_mask=s.legal_action_mask,
        terminated=s.terminated,
        cur_player_id=s.current_player,
        step=s._step_count
    ))(states)

    # Initialize MCTS trees - need batched version
    # Get template from first state
    template = jax.tree_util.tree_map(lambda x: x[0], states)
    single_eval_state = mcts.init(template_embedding=template)

    # Stack to create batch
    eval_states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * batch_size),
        single_eval_state
    )

    step_fn = make_step_fn(env)

    # Vmapped step function
    vmapped_step = jax.vmap(
        lambda s, m, e, k: single_game_step(mcts, step_fn, params, s, m, e, k)
    )

    # Run for num_steps
    total_steps = 0
    step_keys = jax.random.split(key, num_steps)

    for i in range(num_steps):
        # Check active games
        active = ~states.terminated
        active_count = int(jnp.sum(active))
        if active_count == 0:
            break

        # Get keys for this step
        batch_keys = jax.random.split(step_keys[i], batch_size)

        # Run vmapped step
        states, metadatas, eval_states, _ = vmapped_step(
            states, metadatas, eval_states, batch_keys
        )

        total_steps += active_count
        jax.block_until_ready(states.terminated)

    return total_steps


def benchmark(label, batch_size, env, mcts, params, num_steps=10, warmup=1, runs=3):
    """Benchmark vmapped games."""
    print(f"\n{label} (batch_size={batch_size})")
    print("-" * 50)

    key = jax.random.PRNGKey(0)

    # Warmup
    print("  JIT compiling...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(warmup):
        key, subkey = jax.random.split(key)
        run_vmapped_games(env, mcts, params, batch_size, num_steps=3, key=subkey)
    warmup_time = time.perf_counter() - start
    print(f"done ({warmup_time:.1f}s)")

    # Benchmark
    print("  Running...", end=" ", flush=True)
    total_steps = 0
    start = time.perf_counter()
    for _ in range(runs):
        key, subkey = jax.random.split(key)
        steps = run_vmapped_games(env, mcts, params, batch_size, num_steps=num_steps, key=subkey)
        total_steps += steps
        print(f"{steps}", end=" ", flush=True)
    elapsed = time.perf_counter() - start
    print("done")

    steps_per_sec = total_steps / elapsed
    print(f"  Results: {total_steps} steps in {elapsed:.1f}s = {steps_per_sec:.1f} steps/sec")
    return steps_per_sec


def main():
    print("=" * 60)
    print("Vmapped Games Benchmark")
    print("=" * 60)

    # Minimal config
    num_simulations = 10  # Very low for fast JIT
    max_nodes = 50
    num_steps = 10

    print(f"\nConfiguration:")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Steps per game: {num_steps}")

    env = bg.Backgammon(simple_doubles=True, short_game=True)

    model = SimpleNet(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros(34))

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {param_count:,}")

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

    # Test different batch sizes
    for batch_size in [1, 16, 128]:
        results[batch_size] = benchmark(
            f"Vmapped", batch_size, env, mcts, params,
            num_steps=num_steps, warmup=1, runs=2
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for bs, sps in results.items():
        print(f"  Batch {bs:3d}: {sps:.1f} steps/sec")

    if 1 in results and len(results) > 1:
        max_bs = max(results.keys())
        speedup = results[max_bs] / results[1]
        print(f"\n  Batch {max_bs} vs 1: {speedup:.1f}x speedup")


if __name__ == '__main__':
    main()
