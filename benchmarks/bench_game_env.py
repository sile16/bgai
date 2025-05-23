#!/usr/bin/env python3
"""
Game Environment benchmark implementation for testing pure environment step performance.

This benchmark tests the performance of the backgammon environment without any AI evaluators,
providing a baseline for environment step performance using the unified benchmark framework.
"""

import time
import numpy as np
from typing import List, Optional
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
import pgx.backgammon as bg

from benchmarks.benchmark_common import (
    BatchBenchResult,
    DEFAULT_MEMORY_LIMIT_GB,
    DEFAULT_BENCHMARK_DURATION,
    get_memory_usage,
    calculate_memory_percentage,
    format_human_readable,
    random_action_from_mask,
    BaseBenchmark,
)

class GameEnvironmentBenchmark(BaseBenchmark):
    """Benchmark for testing pure game environment step performance."""
    
    def __init__(self):
        super().__init__(
            name="GameEnv",
            description="Pure backgammon environment step performance benchmark"
        )
        self.env = bg.Backgammon(short_game=True)
    
    def step_batch_with_reset(self, key: chex.PRNGKey, states: chex.ArrayTree):
        """Step a batch of environments with automatic reset on termination."""
        step_keys, reset_keys, action_keys = jax.random.split(key, 3)
        batch_step_keys = jax.random.split(step_keys, self.batch_size)
        batch_reset_keys = jax.random.split(reset_keys, self.batch_size)
        batch_action_keys = jax.random.split(action_keys, self.batch_size)
        
        # Sample random actions for each environment
        def sample_action(env_key, state):
            return random_action_from_mask(env_key, state.legal_action_mask)
        
        actions = jax.vmap(sample_action)(batch_action_keys, states)
        
        # Step environments
        def step_single(step_key, state, action):
            return self.env.step(state, action, step_key)
        
        next_states = jax.vmap(step_single)(batch_step_keys, states, actions)
        
        # Check termination
        terminated = next_states.terminated
        
        # Reset terminated environments
        reset_states = jax.vmap(self.env.init)(batch_reset_keys)
        
        # Select between next and reset states
        def where_terminated(next_s, reset_s):
            return jnp.where(
                terminated.reshape(-1, *([1]*(len(next_s.shape)-1))),
                reset_s,
                next_s
            )
        
        final_states = jax.tree_util.tree_map(where_terminated, next_states, reset_states)
        
        return final_states, terminated
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> BatchBenchResult:
        """Run benchmark for a specific batch size."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        self.batch_size = batch_size
        print(f"\nBenchmarking Game Environment: Batch={batch_size} for {max_duration}s", flush=True)
        
        # Initialize
        key = jax.random.PRNGKey(0)
        
        # Initialize states
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, batch_size)
        states = jax.vmap(self.env.init)(init_keys)
        
        # Compile step function
        step_fn = jax.jit(self.step_batch_with_reset)
        
        # Warmup
        print("Compiling and warming up...", flush=True)
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            new_states, _ = step_fn(subkey, states)
            jax.block_until_ready(new_states)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            NUM_WARMUP_ITERATIONS = 4
            states = new_states
            for _ in range(NUM_WARMUP_ITERATIONS):
                key, subkey = jax.random.split(key)
                states, _ = step_fn(subkey, states)
            jax.block_until_ready(states)
            print("Warm-up complete", flush=True)
        except Exception as e:
            print(f"Error warming up: {e}", flush=True)
            raise
        
        # Benchmark loop
        start_time = time.time()
        total_steps = 0
        completed_games = 0
        current_game_steps = np.zeros(batch_size, dtype=np.int32)
        game_lengths = []
        
        # Track memory
        initial_memory_gb = get_memory_usage()
        peak_memory_gb = initial_memory_gb
        
        with tqdm(total=max_duration, desc=f"GameEnv B={batch_size}", unit="s") as pbar:
            while (elapsed_time := time.time() - start_time) < max_duration:
                # Run iteration
                key, subkey = jax.random.split(key)
                try:
                    states, terminated = step_fn(subkey, states)
                    jax.block_until_ready(terminated)
                except Exception as e:
                    print(f"Error during benchmark: {e}", flush=True)
                    break
                
                # Update statistics
                active_mask = ~terminated
                steps_this_iteration = np.sum(active_mask)
                total_steps += steps_this_iteration
                current_game_steps[active_mask] += 1
                
                # Process completed games
                if np.any(terminated):
                    terminated_indices = np.where(terminated)[0]
                    for i in terminated_indices:
                        completed_games += 1
                        game_length = current_game_steps[i]
                        game_lengths.append(game_length)
                        current_game_steps[i] = 0
                
                # Update memory tracking
                current_memory = get_memory_usage()
                peak_memory_gb = max(peak_memory_gb, current_memory)
                
                # Update progress display
                if time.time() - pbar.last_print_t > 0.5:
                    steps_per_sec = total_steps / elapsed_time
                    games_per_sec = completed_games / elapsed_time
                    pbar.set_postfix({
                        "steps/s": f"{format_human_readable(steps_per_sec)}",
                        "games/s": f"{format_human_readable(games_per_sec)}",
                        "mem": f"{peak_memory_gb:.2f}GB"
                    })
        
        # Calculate final metrics
        final_elapsed_time = time.time() - start_time
        steps_per_second = total_steps / final_elapsed_time
        games_per_second = completed_games / final_elapsed_time
        avg_game_length = np.mean(game_lengths) if game_lengths else 0
        median_game_length = np.median(game_lengths) if game_lengths else 0
        min_game_length = np.min(game_lengths) if game_lengths else 0
        max_game_length = np.max(game_lengths) if game_lengths else 0
        memory_percent = calculate_memory_percentage(peak_memory_gb)
        efficiency = steps_per_second / peak_memory_gb
        
        result = BatchBenchResult(
            batch_size=batch_size,
            steps_per_second=steps_per_second,
            games_per_second=games_per_second,
            avg_game_length=avg_game_length,
            median_game_length=median_game_length,
            min_game_length=min_game_length,
            max_game_length=max_game_length,
            memory_usage_gb=peak_memory_gb,
            memory_usage_percent=memory_percent,
            efficiency=efficiency
        )
        
        return result

    def _run_discovery(self, 
                      memory_limit_gb: float, 
                      duration: int, 
                      custom_batch_sizes: Optional[List[int]],
                      verbose: bool) -> List[BatchBenchResult]:
        """Run the discovery process for the game environment benchmark."""
        results = []
        
        # Use custom batch sizes or generate sequence
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            batch_size = 1
            batch_sizes = []
            while batch_size <= 2048:  # Cap at reasonable maximum
                batch_sizes.append(batch_size)
                batch_size *= 2
        
        # Run benchmarks
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size(batch_size, duration)
                results.append(result)
                
                # Check termination conditions  
                if result.memory_usage_percent >= 90:  # 90% memory usage threshold
                    print(f"Memory limit reached: {result.memory_usage_percent:.1f}%", flush=True)
                    break
                    
                if len(results) > 1:
                    perf_improvement = (results[-1].steps_per_second / 
                                      results[-2].steps_per_second - 1.0)
                    if perf_improvement < 0.1:
                        print("Diminishing returns detected (less than 10% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        return results
