#!/usr/bin/env python3
"""
Hybrid CPU/GPU Benchmark with batched games (vmap).

Tests running 1 vs 128 games in parallel using jax.vmap.
- 50 MCTS iterations per move
- 10 moves per game
- ResNet with 256 hidden, 10 blocks
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


# Pre-activation ResNet-V2 block
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
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_dim)(x)
        policy_logits = nn.Dense(self.num_actions)(x)
        # 6-way value head for backgammon outcome distribution
        v = nn.LayerNorm()(x)
        v = nn.relu(v)
        value_logits = nn.Dense(6)(v)
        return policy_logits, value_logits


def create_hybrid_eval_fn_batched(model, params, gpu_device, cpu_device):
    """Create batched eval function: GPU for NN inference on batch of states."""
    params_on_gpu = jax.device_put(params, gpu_device)

    @partial(jax.jit, device=gpu_device)
    def forward_on_gpu_batched(obs_batch):
        # obs_batch shape: (batch_size, obs_dim)
        return jax.vmap(lambda x: model.apply(params_on_gpu, x))(obs_batch)

    def _gpu_inference_batched(obs_batch):
        """Run batched inference on GPU."""
        policy_gpu, value_gpu = forward_on_gpu_batched(obs_batch)
        policy_cpu = jax.device_put(policy_gpu, cpu_device)
        value_cpu = jax.device_put(value_gpu, cpu_device)
        jax.block_until_ready(policy_cpu)
        jax.block_until_ready(value_cpu)
        return policy_cpu, value_cpu

    # The eval_fn for a single state (will be vmapped externally)
    def eval_fn(env_state, params_arg, key):
        obs = env_state.observation
        # For single state, we still go through the batched path
        # but with batch size 1
        policy, value = jax.pure_callback(
            lambda o: tuple(x[0] for x in _gpu_inference_batched(o[None])),
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


def run_single_step(env, mcts, params, state, metadata, eval_state, step_fn, key):
    """Run a single MCTS step for one game."""
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
    new_state, new_metadata = step_fn(state, action, subkey)
    new_eval_state = mcts.step(output.eval_state, action)

    return new_state, new_metadata, new_eval_state, key


def run_batched_games(env, mcts, params, batch_size, max_steps, key):
    """Run batch_size games in parallel using vmap."""
    # Initialize batch of games
    keys = jax.random.split(key, batch_size + 1)
    key = keys[0]
    init_keys = keys[1:]

    # Vectorized init
    states = jax.vmap(env.init)(init_keys)

    # Create batched metadata
    def make_metadata(state):
        return StepMetadata(
            rewards=state.rewards,
            action_mask=state.legal_action_mask,
            terminated=state.terminated,
            cur_player_id=state.current_player,
            step=state._step_count
        )

    metadatas = jax.vmap(make_metadata)(states)

    # Initialize MCTS trees for each game
    # We need to init with a template from one state
    template_state = jax.tree_util.tree_map(lambda x: x[0], states)
    single_eval_state = mcts.init(template_embedding=template_state)

    # Stack to create batched eval_states
    eval_states = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * batch_size),
        single_eval_state
    )

    step_fn = create_step_fn(env)

    # Vmapped step function
    def batched_step(states, metadatas, eval_states, keys):
        """One step across all games."""
        def single_game_step(state, metadata, eval_state, key):
            # Skip if game terminated
            def do_step():
                return run_single_step(env, mcts, params, state, metadata, eval_state, step_fn, key)

            def skip_step():
                return state, metadata, eval_state, key

            return jax.lax.cond(
                state.terminated,
                skip_step,
                do_step
            )

        return jax.vmap(single_game_step)(states, metadatas, eval_states, keys)

    # Run for max_steps
    total_steps = 0
    for step in range(max_steps):
        # Generate keys for this step
        keys = jax.random.split(key, batch_size + 1)
        key = keys[0]
        step_keys = keys[1:]

        # Check how many games are still active
        active = ~states.terminated
        active_count = int(jnp.sum(active))

        if active_count == 0:
            break

        # Run one step for all games
        states, metadatas, eval_states, _ = batched_step(states, metadatas, eval_states, step_keys)
        total_steps += active_count

        # Block for timing accuracy
        jax.block_until_ready(states.terminated)

    return total_steps


def benchmark_batched(label, env, mcts, params, batch_size, max_steps=10, warmup=1, runs=3):
    """Benchmark batched games."""
    print(f"\n{label} (batch_size={batch_size})")
    print("-" * 60)

    key = jax.random.PRNGKey(0)

    # Warmup
    print("  JIT compiling (warmup)...", end=" ", flush=True)
    start = time.perf_counter()
    for _ in range(warmup):
        key, subkey = jax.random.split(key)
        run_batched_games(env, mcts, params, batch_size, max_steps=3, key=subkey)
    warmup_time = time.perf_counter() - start
    print(f"done ({warmup_time:.1f}s)")

    # Benchmark
    print("  Running benchmark...", end=" ", flush=True)
    total_steps = 0
    start = time.perf_counter()
    for i in range(runs):
        key, subkey = jax.random.split(key)
        steps = run_batched_games(env, mcts, params, batch_size, max_steps=max_steps, key=subkey)
        total_steps += steps
        print(f"{steps}", end=" ", flush=True)
    elapsed = time.perf_counter() - start
    print("done")

    steps_per_sec = total_steps / elapsed
    print(f"  Results: {total_steps} total steps in {elapsed:.1f}s = {steps_per_sec:.1f} steps/sec")

    return {'steps_per_sec': steps_per_sec, 'total_steps': total_steps, 'elapsed': elapsed}


def main():
    print("=" * 60)
    print("Batched Games Benchmark (vmap over multiple games)")
    print("=" * 60)

    # Configuration
    num_simulations = 50
    max_nodes = 300
    max_steps = 10

    print(f"\nConfiguration:")
    print(f"  Neural network: ResNet 256 hidden, 10 blocks")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Max nodes: {max_nodes}")
    print(f"  Moves per game: {max_steps}")

    # Create environment
    env = bg.Backgammon(simple_doubles=True, short_game=True)

    # Create network
    model = ResNetTurboZero(num_actions=env.num_actions)
    key = jax.random.PRNGKey(42)
    sample_obs = jnp.zeros(34)
    params = model.init(key, sample_obs)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model parameters: {param_count:,}")

    results = {}

    # ================================================================
    # Test batch_size = 1
    # ================================================================
    print("\n" + "=" * 60)
    print("Batch Size = 1")
    print("=" * 60)

    with jax.default_device(_cpu_device):
        cpu_params = jax.device_put(params, _cpu_device)
        cpu_eval_fn = create_cpu_eval_fn(model, cpu_params)

        mcts_1 = StochasticMCTS(
            eval_fn=cpu_eval_fn,
            action_selector=PUCTSelector(),
            stochastic_action_probs=env.stochastic_action_probs,
            branching_factor=env.num_actions,
            max_nodes=max_nodes,
            num_iterations=num_simulations,
        )

        results['batch_1_cpu'] = benchmark_batched(
            "CPU-only", env, mcts_1, cpu_params, batch_size=1, max_steps=max_steps
        )

    if _has_gpu:
        hybrid_eval_fn = create_hybrid_eval_fn_batched(model, params, _gpu_device, _cpu_device)

        with jax.default_device(_cpu_device):
            mcts_1_hybrid = StochasticMCTS(
                eval_fn=hybrid_eval_fn,
                action_selector=PUCTSelector(),
                stochastic_action_probs=env.stochastic_action_probs,
                branching_factor=env.num_actions,
                max_nodes=max_nodes,
                num_iterations=num_simulations,
            )

            results['batch_1_hybrid'] = benchmark_batched(
                "Hybrid (CPU + GPU)", env, mcts_1_hybrid, params, batch_size=1, max_steps=max_steps
            )

    # ================================================================
    # Test batch_size = 128
    # ================================================================
    print("\n" + "=" * 60)
    print("Batch Size = 128")
    print("=" * 60)

    with jax.default_device(_cpu_device):
        results['batch_128_cpu'] = benchmark_batched(
            "CPU-only", env, mcts_1, cpu_params, batch_size=128, max_steps=max_steps
        )

    if _has_gpu:
        with jax.default_device(_cpu_device):
            results['batch_128_hybrid'] = benchmark_batched(
                "Hybrid (CPU + GPU)", env, mcts_1_hybrid, params, batch_size=128, max_steps=max_steps
            )

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nSteps per second:")
    print(f"  Batch=1   CPU:    {results['batch_1_cpu']['steps_per_sec']:.1f}")
    if 'batch_1_hybrid' in results:
        print(f"  Batch=1   Hybrid: {results['batch_1_hybrid']['steps_per_sec']:.1f}")
    print(f"  Batch=128 CPU:    {results['batch_128_cpu']['steps_per_sec']:.1f}")
    if 'batch_128_hybrid' in results:
        print(f"  Batch=128 Hybrid: {results['batch_128_hybrid']['steps_per_sec']:.1f}")

    print("\nSpeedups:")
    cpu_speedup = results['batch_128_cpu']['steps_per_sec'] / results['batch_1_cpu']['steps_per_sec']
    print(f"  Batch 128 vs 1 (CPU):    {cpu_speedup:.1f}x")

    if 'batch_128_hybrid' in results and 'batch_1_hybrid' in results:
        hybrid_speedup = results['batch_128_hybrid']['steps_per_sec'] / results['batch_1_hybrid']['steps_per_sec']
        print(f"  Batch 128 vs 1 (Hybrid): {hybrid_speedup:.1f}x")

        hybrid_vs_cpu = results['batch_128_hybrid']['steps_per_sec'] / results['batch_128_cpu']['steps_per_sec']
        print(f"  Hybrid vs CPU (Batch=128): {hybrid_vs_cpu:.1f}x")


if __name__ == '__main__':
    main()
