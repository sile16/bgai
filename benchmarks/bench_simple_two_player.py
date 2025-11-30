#!/usr/bin/env python3
"""
Simplified Two-Player benchmark for testing simultaneous episode performance.

This benchmark uses simple heuristic evaluators to avoid complex JAX tracing issues
while still demonstrating the framework for two-player evaluation benchmarking.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
from functools import partial

from bgai.bgevaluators import backgammon_pip_count_eval
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


class SimpleTwoPlayerBenchmark(BaseBenchmark):
    """Simplified benchmark for testing two-player evaluation performance."""
    
    def __init__(self):
        super().__init__(
            name="SimpleTwoPlayer",
            description="Simplified two-player heuristic evaluation benchmark"
        )
        
        # Import backgammon environment
        import pgx.backgammon as bg
        self.bg_env = bg.Backgammon(short_game=True)
    
    def play_simple_game(self, key: chex.PRNGKey) -> Tuple[int, int]:
        """Play a simple two-player game using heuristic evaluation.
        
        Returns:
            Tuple of (total_steps, winner)
        """
        game_key, p1_key, p2_key = jax.random.split(key, 3)
        
        # Initialize game
        state = self.bg_env.init(game_key)
        steps = 0
        max_steps = 500
        
        def game_loop_cond(loop_state):
            state, steps, _ = loop_state
            return jnp.logical_and(jnp.logical_not(state.terminated), steps < max_steps)
        
        def game_loop_body(loop_state):
            state, steps, key = loop_state
            
            # Use pip count evaluation for action selection
            policy_logits, _ = backgammon_pip_count_eval(state, jnp.array([]), jax.random.PRNGKey(0))
            
            # Select best legal action
            legal_logits = jnp.where(state.legal_action_mask, policy_logits, -jnp.inf)
            action = jnp.argmax(legal_logits)
            
            # Step environment using conditional for stochastic vs deterministic
            step_key = jax.random.split(key, 2)[1]
            
            def stochastic_step(operand):
                s, a, _ = operand
                return self.bg_env.stochastic_step(s, a)
            
            def deterministic_step(operand):
                s, a, k = operand
                return self.bg_env.step(s, a, k)
            
            new_state = jax.lax.cond(
                state._is_stochastic,
                stochastic_step,
                deterministic_step,
                (state, action, step_key)
            )
            
            return new_state, steps + 1, step_key
        
        # Run game loop
        final_state, final_steps, _ = jax.lax.while_loop(
            game_loop_cond,
            game_loop_body,
            (state, steps, game_key)
        )
        
        # Determine winner (0 or 1)
        winner = jnp.where(final_state.rewards[0] > final_state.rewards[1], 0, 1)
        return final_steps, winner
    
    def run_episode_batch(self, key: chex.PRNGKey, batch_size: int) -> Tuple[chex.Array, int]:
        """Run a batch of simple two-player games."""
        # Split keys for each episode
        episode_keys = jax.random.split(key, batch_size)
        
        # Vectorize the game function
        vectorized_game = jax.vmap(self.play_simple_game)
        
        # Run games in parallel
        steps_per_game, winners = vectorized_game(episode_keys)
        
        # Calculate total steps
        total_steps = jnp.sum(steps_per_game)
        
        return winners, total_steps
    
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> BatchBenchResult:
        """Run benchmark for a specific batch size."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        print(f"\nBenchmarking Simple Two-Player: Batch={batch_size} for {max_duration}s", flush=True)
        
        # Initialize
        key = jax.random.PRNGKey(0)
        
        # Compile episode runner
        run_batch_fn = jax.jit(self.run_episode_batch, static_argnums=(1,))
        
        # Warmup compilation
        print("Compiling and warming up...", flush=True)
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            winners, total_steps = run_batch_fn(subkey, batch_size)
            jax.block_until_ready(winners)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            NUM_WARMUP_ITERATIONS = 2
            for _ in range(NUM_WARMUP_ITERATIONS):
                key, subkey = jax.random.split(key)
                winners, total_steps = run_batch_fn(subkey, batch_size)
                jax.block_until_ready(winners)
            print("Warm-up complete", flush=True)
        except Exception as e:
            print(f"Error warming up: {e}", flush=True)
            raise
        
        # Benchmark loop
        start_time = time.time()
        total_episodes = 0
        total_steps = 0
        episode_lengths = []
        
        # Track memory
        initial_memory_gb = get_memory_usage()
        peak_memory_gb = initial_memory_gb
        
        with tqdm(total=max_duration, desc=f"SimpleTwoPlayer B={batch_size}", unit="s") as pbar:
            while (elapsed_time := time.time() - start_time) < max_duration:
                # Run batch of episodes
                key, subkey = jax.random.split(key)
                try:
                    winners, steps_this_batch = run_batch_fn(subkey, batch_size)
                    jax.block_until_ready(winners)
                except Exception as e:
                    print(f"Error during benchmark: {e}", flush=True)
                    break
                
                # Update statistics
                episodes_completed = batch_size
                total_episodes += episodes_completed
                steps_this_batch_int = int(steps_this_batch.item())
                total_steps += steps_this_batch_int
                
                # Track episode lengths
                avg_length_this_batch = steps_this_batch_int // batch_size
                for _ in range(episodes_completed):
                    episode_lengths.append(avg_length_this_batch)
                
                # Update memory tracking
                current_memory = get_memory_usage()
                peak_memory_gb = max(peak_memory_gb, current_memory)
                
                # Update progress display
                if time.time() - pbar.last_print_t > 0.5:
                    episodes_per_sec = total_episodes / elapsed_time
                    steps_per_sec = total_steps / elapsed_time
                    pbar.set_postfix({
                        "episodes/s": f"{format_human_readable(episodes_per_sec)}",
                        "steps/s": f"{format_human_readable(steps_per_sec)}",
                        "mem": f"{peak_memory_gb:.2f}GB"
                    })
        
        # Calculate final metrics
        final_elapsed_time = time.time() - start_time
        episodes_per_second = total_episodes / final_elapsed_time
        steps_per_second = total_steps / final_elapsed_time
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
        median_episode_length = np.median(episode_lengths) if episode_lengths else 0
        min_episode_length = np.min(episode_lengths) if episode_lengths else 0
        max_episode_length = np.max(episode_lengths) if episode_lengths else 0
        memory_percent = calculate_memory_percentage(peak_memory_gb)
        efficiency = steps_per_second / peak_memory_gb
        
        result = BatchBenchResult(
            batch_size=batch_size,
            steps_per_second=steps_per_second,
            games_per_second=episodes_per_second,
            avg_game_length=avg_episode_length,
            median_game_length=median_episode_length,
            min_game_length=min_episode_length,
            max_game_length=max_episode_length,
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
        """Run the discovery process for the simple two-player benchmark."""
        results = []
        
        # Use custom batch sizes or generate sequence
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
        # Run benchmarks
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size(batch_size, duration)
                results.append(result)
                
                # Check termination conditions  
                if result.memory_usage_percent >= 90:
                    print(f"Memory limit reached: {result.memory_usage_percent:.1f}%", flush=True)
                    break
                    
                if len(results) > 1:
                    perf_improvement = (results[-1].games_per_second / 
                                      results[-2].games_per_second - 1.0)
                    if perf_improvement < 0.05:
                        print("Diminishing returns detected (less than 5% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        return results
