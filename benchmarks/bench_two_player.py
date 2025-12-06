#!/usr/bin/env python3
"""
Two-Player Baseline benchmark implementation for testing simultaneous episode performance.

This benchmark tests the performance of running multiple simultaneous two-player episodes
using the TurboZero baseline evaluator, providing insights into optimal batch sizes for evaluation.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
from functools import partial

# TurboZero imports
from core.common import two_player_game
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.networks.mlp import MLP, MLPConfig
from core.types import EnvInitFn, EnvStepFn, StepMetadata

# Local imports  
from bgai.bgevaluators import backgammon_pip_count_eval
from benchmarks.benchmark_common import (
    BatchBenchResult,
    DEFAULT_MEMORY_LIMIT_GB,
    DEFAULT_BENCHMARK_DURATION,
    get_memory_usage,
    calculate_memory_percentage,
    format_human_readable,
    BaseBenchmark,
)


class TwoPlayerBaselineBenchmark(BaseBenchmark):
    """Benchmark for testing two-player baseline evaluation performance."""
    
    def __init__(self):
        super().__init__(
            name="TwoPlayerBaseline", 
            description="Two-player StochasticMCTS benchmark: Neural Network vs Heuristic evaluators"
        )
        
        # Import backgammon environment
        import pgx.backgammon as bg
        self.bg_env = bg.Backgammon(short_game=True)
        
        # Create step and init functions
        self.env_step_fn = self._create_step_fn()
        self.env_init_fn = self._create_init_fn()
        
        # Create Neural Network for NN evaluator with 6-way value head
        mlp_config = MLPConfig(
            hidden_dims=[256, 256, 256],
            policy_head_out_size=self.bg_env.num_actions,
            value_head_out_size=6  # 6-way outcome distribution for backgammon
        )
        self.network = MLP(mlp_config)
        
        # Create evaluation function for neural network
        self.nn_eval_fn = make_nn_eval_fn(self.network, self._create_state_to_nn_input_fn())
        
        # Create action selector
        action_selector = PUCTSelector(c=1.0)
        
        # Create StochasticMCTS evaluator with Neural Network
        self.nn_evaluator = StochasticMCTS(
            eval_fn=self.nn_eval_fn,
            action_selector=action_selector,
            stochastic_action_probs=jnp.ones(self.bg_env.num_actions) / self.bg_env.num_actions,
            branching_factor=self.bg_env.num_actions,
            max_nodes=500,
            num_iterations=50,
            discount=-1.0,  # Two-player game
            temperature=1.0
        )
        
        # Create StochasticMCTS evaluator with heuristic (pip count)
        self.heuristic_evaluator = StochasticMCTS(
            eval_fn=backgammon_pip_count_eval,
            action_selector=action_selector,
            stochastic_action_probs=jnp.ones(self.bg_env.num_actions) / self.bg_env.num_actions,
            branching_factor=self.bg_env.num_actions,
            max_nodes=500,
            num_iterations=50,
            discount=-1.0,  # Two-player game
            temperature=1.0
        )
    
    def _create_step_fn(self) -> EnvStepFn:
        """Create the environment step function."""
        def step_fn(state, action, key):
            def stochastic_branch(operand):
                s, a, _ = operand
                return self.bg_env.stochastic_step(s, a)

            def deterministic_branch(operand):
                s, a, k = operand
                return self.bg_env.step(s, a, k)

            new_state = jax.lax.cond(
                state._is_stochastic,
                stochastic_branch,
                deterministic_branch,
                (state, action, key)
            )
            
            # Create metadata for TurboZero compatibility
            metadata = StepMetadata(
                rewards=new_state.rewards,
                action_mask=new_state.legal_action_mask,
                terminated=new_state.terminated,
                cur_player_id=new_state.current_player,
                step=new_state._step_count
            )
            
            return new_state, metadata
        
        return step_fn
    
    def _create_init_fn(self) -> EnvInitFn:
        """Create the environment initialization function."""
        def init_fn(key):
            state = self.bg_env.init(key)
            metadata = StepMetadata(
                rewards=state.rewards,
                action_mask=state.legal_action_mask,
                terminated=state.terminated,
                cur_player_id=state.current_player,
                step=state._step_count
            )
            return state, metadata
        
        return init_fn
    
    def run_episode_batch(self, key: chex.PRNGKey, batch_size: int, use_nn_params: bool = False) -> Tuple[chex.Array, int]:
        """Run a batch of two-player episodes.
        
        Args:
            key: Random key
            batch_size: Number of episodes to run
            use_nn_params: Whether to use neural network parameters (vs random initialization)
        """
        # Split keys for each episode
        episode_keys = jax.random.split(key, batch_size)
        
        # Initialize neural network parameters if needed
        if use_nn_params:
            # Initialize with dummy input
            dummy_state = self.bg_env.init(jax.random.PRNGKey(0))
            dummy_obs = self._create_state_to_nn_input_fn()(dummy_state)
            nn_params = self.network.init(jax.random.PRNGKey(0), dummy_obs[None, ...], train=False)
            params_1 = nn_params
        else:
            # Use empty params for heuristic evaluator
            params_1 = jnp.array([])
        
        params_2 = jnp.array([])  # Heuristic always uses empty params
        
        # Create the game function
        game_fn = partial(
            two_player_game,
            evaluator_1=self.nn_evaluator,      # Neural Network + StochasticMCTS
            evaluator_2=self.heuristic_evaluator, # Heuristic + StochasticMCTS
            params_1=params_1,
            params_2=params_2,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=1000  # Maximum steps per episode
        )
        
        # Run episodes in parallel
        results, frames, p_ids = jax.vmap(game_fn)(episode_keys)
        
        # Count total steps across all episodes
        # Since we don't have direct step counts, estimate based on completion
        total_steps = batch_size * 40  # Rough estimate for MCTS backgammon games
        
        return results, total_steps
    
    def _create_state_to_nn_input_fn(self):
        """Create function to convert environment state to neural network input."""
        def state_to_nn_input(state):
            # Simple flattening of the board for MLP
            # In practice, you'd use proper feature extraction
            board = state._board.flatten()
            # Pad or truncate to fixed size
            if len(board) < 256:
                board = jnp.pad(board, (0, 256 - len(board)))
            else:
                board = board[:256]
            return board.astype(jnp.float32)
        
        return state_to_nn_input
    
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> BatchBenchResult:
        """Run benchmark for a specific batch size."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        print(f"\nBenchmarking Two-Player Baseline: Batch={batch_size} for {max_duration}s", flush=True)
        
        # Initialize
        key = jax.random.PRNGKey(0)
        
        # Compile episode runner
        run_batch_fn = jax.jit(self.run_episode_batch, static_argnums=(1, 2))  # batch_size and use_nn_params are static
        
        # Warmup compilation
        print("Compiling and warming up...", flush=True)
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            results, total_steps = run_batch_fn(subkey, batch_size, False)  # Start with heuristic only
            jax.block_until_ready(results)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            NUM_WARMUP_ITERATIONS = 2  # Fewer warmups for episodes
            for _ in range(NUM_WARMUP_ITERATIONS):
                key, subkey = jax.random.split(key)
                results, total_steps = run_batch_fn(subkey, batch_size, False)
                jax.block_until_ready(results)
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
        
        with tqdm(total=max_duration, desc=f"TwoPlayer B={batch_size}", unit="s") as pbar:
            while (elapsed_time := time.time() - start_time) < max_duration:
                # Run batch of episodes
                key, subkey = jax.random.split(key)
                try:
                    results, steps_this_batch = run_batch_fn(subkey, batch_size, False)  # Use heuristic evaluator
                    jax.block_until_ready(results)
                except Exception as e:
                    print(f"Error during benchmark: {e}", flush=True)
                    break
                
                # Update statistics
                episodes_completed = batch_size
                total_episodes += episodes_completed
                total_steps += steps_this_batch
                
                # Track episode lengths (estimated)
                estimated_episode_length = steps_this_batch // batch_size
                for _ in range(episodes_completed):
                    episode_lengths.append(estimated_episode_length)
                
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
        """Run the discovery process for the two-player baseline benchmark."""
        results = []
        
        # Use custom batch sizes or generate sequence
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            # Start with smaller batch sizes for episodes since they're more expensive
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # Cap at reasonable maximum for episodes
        
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
                    perf_improvement = (results[-1].games_per_second / 
                                      results[-2].games_per_second - 1.0)
                    if perf_improvement < 0.05:  # Lower threshold for episode performance
                        print("Diminishing returns detected (less than 5% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        return results