#!/usr/bin/env python3
"""
Test different training batch sizes to find optimal GPU utilization.
Measures memory usage and GPU utilization for various batch sizes.
"""

import os
import sys
import time
import argparse

# Set memory fraction before JAX import
def run_test(batch_size: int, memory_fraction: float, num_steps: int = 50):
    """Run training test with specified batch size."""
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import flax.linen as nn
    from flax.training.train_state import TrainState
    import pickle
    import glob

    print(f"\n{'='*60}")
    print(f"TRAINING BATCH SIZE TEST")
    print(f"  Batch size: {batch_size}")
    print(f"  Memory fraction: {memory_fraction}")
    print(f"  Num steps: {num_steps}")
    print(f"{'='*60}\n")

    # Network definition (same as training_worker.py)
    class ResidualDenseBlock(nn.Module):
        features: int
        @nn.compact
        def __call__(self, x):
            residual = x
            x = nn.Dense(self.features)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            x = nn.Dense(self.features)(x)
            x = nn.LayerNorm()(x)
            return nn.relu(x + residual)

    class ResNetTurboZero(nn.Module):
        num_actions: int = 156
        num_hidden: int = 256
        num_blocks: int = 6
        value_head_out_size: int = 6
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.num_hidden)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)
            for _ in range(self.num_blocks):
                x = ResidualDenseBlock(features=self.num_hidden)(x)
            policy_logits = nn.Dense(self.num_actions)(x)
            value = nn.Dense(self.value_head_out_size)(x)
            return policy_logits, value

    print("Initializing network...")
    network = ResNetTurboZero()

    # Load params from checkpoint and get input dimension
    checkpoint_dir = '/home/sile/github/bgai/checkpoints'
    checkpoint_files = sorted(glob.glob(f'{checkpoint_dir}/ckpt_*.pkl'))
    input_dim = 86  # Default from checkpoint
    if checkpoint_files:
        with open(checkpoint_files[-1], 'rb') as f:
            ckpt = pickle.load(f)
        params = ckpt.get('params', ckpt)
        # Get input dim from checkpoint
        input_dim = params['Dense_0']['kernel'].shape[0]
        print(f"Loaded params from {checkpoint_files[-1]}")
        print(f"Input dimension: {input_dim}")
    else:
        # Initialize random params
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, input_dim))
        params = network.init(rng, dummy_input)['params']
        print("Using random params (no checkpoint found)")

    # Create optimizer and train state
    optimizer = optax.adam(learning_rate=0.0003)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
    )

    # Define loss function (simplified)
    def loss_fn(params, observations, policy_targets, value_targets, policy_masks):
        policy_logits, value_logits = network.apply({'params': params}, observations)

        # Mask invalid actions
        policy_logits = jnp.where(policy_masks, policy_logits, jnp.finfo(jnp.float32).min)

        # Policy loss
        policy_loss = optax.softmax_cross_entropy(policy_logits, policy_targets).mean()

        # Value loss
        value_log_probs = jax.nn.log_softmax(value_logits, axis=-1)
        value_loss = -(value_targets * value_log_probs).sum(axis=-1).mean()

        return policy_loss + value_loss

    # JIT compile training step
    @jax.jit
    def train_step(state, observations, policy_targets, value_targets, policy_masks):
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, observations, policy_targets, value_targets, policy_masks
        )
        state = state.apply_gradients(grads=grads)
        return state, loss

    print("Creating synthetic batch data...")
    # Create synthetic batch data (same shapes as real data)
    rng = jax.random.PRNGKey(42)
    observations = jax.random.normal(rng, (batch_size, input_dim))
    policy_targets = jax.nn.softmax(jax.random.normal(rng, (batch_size, 156)))
    value_targets = jax.nn.softmax(jax.random.normal(rng, (batch_size, 6)))
    policy_masks = jax.random.uniform(rng, (batch_size, 156)) > 0.5

    print("JIT compiling training step...")
    jit_start = time.time()
    train_state, loss = train_step(train_state, observations, policy_targets, value_targets, policy_masks)
    _ = float(loss)  # Block until done
    print(f"JIT compilation took {time.time() - jit_start:.2f}s")

    # Get initial memory stats
    try:
        mem_stats = jax.local_devices()[0].memory_stats()
        if mem_stats:
            initial_bytes = mem_stats.get('bytes_in_use', 0)
            print(f"Initial memory: {initial_bytes / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Could not get memory stats: {e}")
        initial_bytes = 0

    # Run training steps and measure time
    print(f"\nRunning {num_steps} training steps...")
    times = []

    for step in range(num_steps):
        start = time.perf_counter()
        train_state, loss = train_step(train_state, observations, policy_targets, value_targets, policy_masks)
        _ = float(loss)  # Block until done
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if step % 10 == 0:
            print(f"  Step {step}: loss={float(loss):.4f}, time={elapsed*1000:.1f}ms")

    # Get final memory stats
    try:
        mem_stats = jax.local_devices()[0].memory_stats()
        if mem_stats:
            final_bytes = mem_stats.get('bytes_in_use', 0)
            peak_bytes = mem_stats.get('peak_bytes_in_use', 0)
        else:
            final_bytes = peak_bytes = 0
    except:
        final_bytes = peak_bytes = 0

    # Calculate statistics
    times_arr = np.array(times)
    avg_time = times_arr.mean()
    std_time = times_arr.std()
    samples_per_sec = batch_size / avg_time

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Batch size:      {batch_size}")
    print(f"Avg step time:   {avg_time*1000:.2f}ms")
    print(f"Std step time:   {std_time*1000:.2f}ms")
    print(f"Samples/sec:     {samples_per_sec:.0f}")
    print(f"Memory in use:   {final_bytes / 1024**3:.2f} GB")
    print(f"Peak memory:     {peak_bytes / 1024**3:.2f} GB")

    return {
        'batch_size': batch_size,
        'avg_time_ms': avg_time * 1000,
        'samples_per_sec': samples_per_sec,
        'memory_gb': final_bytes / 1024**3,
        'peak_memory_gb': peak_bytes / 1024**3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--memory-fraction', type=float, default=0.90)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--sweep', action='store_true', help='Test multiple batch sizes')
    args = parser.parse_args()

    if args.sweep:
        # Test multiple batch sizes
        batch_sizes = [256, 512, 1024, 2048, 4096]
        results = []

        for bs in batch_sizes:
            try:
                print(f"\n\n{'#'*60}")
                print(f"TESTING BATCH SIZE: {bs}")
                print(f"{'#'*60}")

                # Fork to get clean JAX state for each test
                result = run_test(bs, args.memory_fraction, args.num_steps)
                results.append(result)
            except Exception as e:
                print(f"ERROR with batch_size={bs}: {e}")
                results.append({'batch_size': bs, 'error': str(e)})

        print(f"\n\n{'='*60}")
        print("SWEEP SUMMARY")
        print(f"{'='*60}")
        for r in results:
            if 'error' in r:
                print(f"  {r['batch_size']:5d}: ERROR - {r['error']}")
            else:
                print(f"  {r['batch_size']:5d}: {r['samples_per_sec']:6.0f} samples/sec, {r['peak_memory_gb']:.2f} GB peak")
    else:
        run_test(args.batch_size, args.memory_fraction, args.num_steps)
