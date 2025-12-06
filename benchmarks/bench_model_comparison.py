#!/usr/bin/env python3
"""
Model Comparison benchmark for testing different evaluators head-to-head.

This benchmark compares neural network evaluators vs heuristic evaluators using MCTS
with specific configurations (1000 nodes, 300 iterations) and tracks both games/s
and average reward metrics for different batch sizes.
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


class ModelComparisonBenchmark(BaseBenchmark):
    """Benchmark for comparing different model evaluators in head-to-head play."""
    
    def __init__(self, model1_name: str = "neural_net", model2_name: str = "heuristic", 
                 num_simulations: int = 300, max_nodes: int = 1000,
                 model1_params_path: Optional[str] = None, model2_params_path: Optional[str] = None):
        super().__init__(
            name=f"ModelComparison_{model1_name}_vs_{model2_name}",
            description=f"Model comparison: {model1_name} vs {model2_name} with {num_simulations} sims, {max_nodes} nodes"
        )
        
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.num_simulations = num_simulations
        self.max_nodes = max_nodes
        self.model1_params_path = model1_params_path
        self.model2_params_path = model2_params_path
        
        # Import backgammon environment
        import pgx.backgammon as bg
        self.bg_env = bg.Backgammon(short_game=True)
        
        # Create step and init functions
        self.env_step_fn = self._create_step_fn()
        self.env_init_fn = self._create_init_fn()
        
        # Create evaluators and initialize parameters
        self.evaluator1, self.evaluator2 = self._create_evaluators()
        self.params1, self.params2 = self._initialize_parameters()
    
    def get_profile_params(self) -> Dict[str, Any]:
        """Return parameters that uniquely identify this benchmark configuration."""
        return {
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "num_simulations": self.num_simulations,
            "max_nodes": self.max_nodes
        }
    
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
    
    def _create_state_to_nn_input_fn(self):
        """Create function to convert state to neural network input."""
        def state_to_nn_input(state):
            return state.observation
        return state_to_nn_input
    
    def _create_evaluators(self):
        """Create the two evaluators to compare."""
        # Create action selector
        action_selector = PUCTSelector(c=1.0)
        
        # Create neural network if needed (reusing notebook pattern)
        self.network = None
        if self.model1_name == "neural_net" or self.model1_name == "resnet_10layer":
            import flax.linen as nn
            
            # Define a dense residual block for vector inputs (from notebook)
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

            # ResNet-style network (from notebook)
            class ResNetTurboZero(nn.Module):
                num_actions: int
                num_hidden: int = 128
                num_blocks: int = 5  # More blocks for 10-layer network

                @nn.compact
                def __call__(self, x, train: bool = False):
                    # Initial projection
                    x = nn.Dense(self.num_hidden)(x)
                    x = nn.LayerNorm()(x)
                    x = nn.relu(x)

                    # Process through residual blocks
                    for _ in range(self.num_blocks):
                        x = ResidualDenseBlock(self.num_hidden)(x)

                    # Policy head
                    policy_logits = nn.Dense(self.num_actions)(x)

                    # 6-way value head for backgammon outcome distribution
                    value_logits = nn.Dense(6)(x)
                    return policy_logits, value_logits

            # Create network instance and store it
            self.network = ResNetTurboZero(self.bg_env.num_actions, num_hidden=128, num_blocks=5)
            
            # Create evaluation function for neural network (from notebook)
            nn_eval_fn = make_nn_eval_fn(self.network, self._create_state_to_nn_input_fn())
            
            evaluator1 = StochasticMCTS(
                eval_fn=nn_eval_fn,
                action_selector=action_selector,
                stochastic_action_probs=jnp.ones(self.bg_env.num_actions) / self.bg_env.num_actions,
                branching_factor=self.bg_env.num_actions,
                max_nodes=self.max_nodes,
                num_iterations=self.num_simulations,
                discount=-1.0,  # Two-player game
                temperature=0.0  # Deterministic for evaluation
            )
        else:
            # Use heuristic for model1
            evaluator1 = StochasticMCTS(
                eval_fn=backgammon_pip_count_eval,
                action_selector=action_selector,
                stochastic_action_probs=jnp.ones(self.bg_env.num_actions) / self.bg_env.num_actions,
                branching_factor=self.bg_env.num_actions,
                max_nodes=self.max_nodes,
                num_iterations=self.num_simulations,
                discount=-1.0,  # Two-player game
                temperature=0.0
            )
        
        # Model 2: Always use heuristic (pip count)
        evaluator2 = StochasticMCTS(
            eval_fn=backgammon_pip_count_eval,
            action_selector=action_selector,
            stochastic_action_probs=jnp.ones(self.bg_env.num_actions) / self.bg_env.num_actions,
            branching_factor=self.bg_env.num_actions,
            max_nodes=self.max_nodes,
            num_iterations=self.num_simulations,
            discount=-1.0,  # Two-player game
            temperature=0.0
        )
        
        return evaluator1, evaluator2
    
    def _initialize_parameters(self):
        """Initialize or load parameters for both models."""
        import orbax.checkpoint as ocp
        
        # Initialize parameters for model 1
        if self.model1_name == "neural_net" or self.model1_name == "resnet_10layer":
            if self.model1_params_path:
                # Load parameters from checkpoint file
                print(f"Loading model 1 parameters from: {self.model1_params_path}")
                try:
                    # Create checkpoint manager for loading
                    checkpoint_manager = ocp.CheckpointManager(
                        directory=self.model1_params_path,
                        checkpointers={
                            'state': ocp.StandardCheckpointer(),
                        }
                    )
                    
                    # Create dummy parameters to match structure for loading
                    dummy_state = self.bg_env.init(jax.random.PRNGKey(0))
                    dummy_obs = dummy_state.observation[None, ...]  # Add batch dimension
                    dummy_params = self.network.init(jax.random.PRNGKey(0), dummy_obs, train=False)
                    
                    # Load checkpoint (get latest step)
                    latest_step = checkpoint_manager.latest_step()
                    if latest_step is not None:
                        params1 = checkpoint_manager.restore(latest_step, items=dummy_params)
                        print(f"Loaded parameters from step {latest_step}")
                    else:
                        print("No checkpoint found, using random parameters")
                        params1 = dummy_params
                        
                except Exception as e:
                    print(f"Error loading parameters: {e}, using random parameters")
                    # Fall back to random initialization
                    dummy_state = self.bg_env.init(jax.random.PRNGKey(0))
                    dummy_obs = dummy_state.observation[None, ...]  # Add batch dimension
                    params1 = self.network.init(jax.random.PRNGKey(42), dummy_obs, train=False)
            else:
                # Initialize random parameters
                print("Initializing random parameters for model 1")
                dummy_state = self.bg_env.init(jax.random.PRNGKey(0))
                dummy_obs = dummy_state.observation[None, ...]  # Add batch dimension  
                params1 = self.network.init(jax.random.PRNGKey(42), dummy_obs, train=False)
        else:
            # Heuristic model doesn't need parameters
            params1 = {}
        
        # Initialize parameters for model 2 (similar logic)
        if self.model2_name == "neural_net" or self.model2_name == "resnet_10layer":
            if self.model2_params_path:
                # Load parameters from checkpoint file (similar to model 1)
                print(f"Loading model 2 parameters from: {self.model2_params_path}")
                # Implementation similar to model 1...
                params2 = {}  # Placeholder for now
            else:
                # Use different random seed for model 2
                print("Initializing random parameters for model 2")
                dummy_state = self.bg_env.init(jax.random.PRNGKey(0))
                dummy_obs = dummy_state.observation[None, ...]
                params2 = self.network.init(jax.random.PRNGKey(123), dummy_obs, train=False) 
        else:
            # Heuristic model doesn't need parameters
            params2 = {}
            
        return params1, params2
    
    def run_episode_batch(self, key: chex.PRNGKey, batch_size: int) -> Tuple[chex.Array, chex.Array, int]:
        """Run a batch of head-to-head episodes."""
        # Split keys for each episode
        episode_keys = jax.random.split(key, batch_size)
        
        # Create the game function with partial using initialized parameters
        game_fn = partial(
            two_player_game,
            evaluator_1=self.evaluator1,
            evaluator_2=self.evaluator2,
            params_1=self.params1,
            params_2=self.params2,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=500  # Maximum steps per episode
        )
        
        # Run games in parallel
        results, frames, p_ids = jax.vmap(game_fn)(episode_keys)
        
        # Extract rewards for each player
        player1_rewards = results[:, 0]  # Rewards for model 1
        player2_rewards = results[:, 1]  # Rewards for model 2
        
        # Estimate total steps (since we don't have direct step counts)
        total_steps = batch_size * 40  # Rough estimate for MCTS backgammon games
        
        return player1_rewards, player2_rewards, jnp.array(total_steps)
    
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> Dict[str, Any]:
        """Run benchmark for a specific batch size and return detailed results."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        print(f"\nBenchmarking Model Comparison: Batch={batch_size} for {max_duration}s", flush=True)
        print(f"Model 1 ({self.model1_name}) vs Model 2 ({self.model2_name})", flush=True)
        print(f"MCTS Config: {self.num_simulations} simulations, {self.max_nodes} max nodes", flush=True)
        
        # Initialize
        key = jax.random.PRNGKey(0)
        
        # Compile episode runner
        run_batch_fn = jax.jit(self.run_episode_batch, static_argnums=(1,))
        
        # Warmup compilation
        print("Compiling and warming up...", flush=True)
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            p1_rewards, p2_rewards, total_steps = run_batch_fn(subkey, batch_size)
            jax.block_until_ready((p1_rewards, p2_rewards))
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            NUM_WARMUP_ITERATIONS = 2
            for _ in range(NUM_WARMUP_ITERATIONS):
                key, subkey = jax.random.split(key)
                p1_rewards, p2_rewards, total_steps = run_batch_fn(subkey, batch_size)
                jax.block_until_ready((p1_rewards, p2_rewards))
            print("Warm-up complete", flush=True)
        except Exception as e:
            print(f"Error warming up: {e}", flush=True)
            raise
        
        # Benchmark loop
        start_time = time.time()
        total_episodes = 0
        total_steps = 0
        p1_total_reward = 0.0
        p2_total_reward = 0.0
        episode_lengths = []
        
        # Track memory
        initial_memory_gb = get_memory_usage()
        peak_memory_gb = initial_memory_gb
        
        with tqdm(total=max_duration, desc=f"ModelComp B={batch_size}", unit="s") as pbar:
            while (elapsed_time := time.time() - start_time) < max_duration:
                # Run batch of episodes
                key, subkey = jax.random.split(key)
                try:
                    p1_rewards, p2_rewards, steps_this_batch = run_batch_fn(subkey, batch_size)
                    jax.block_until_ready((p1_rewards, p2_rewards))
                except Exception as e:
                    print(f"Error during benchmark: {e}", flush=True)
                    break
                
                # Update statistics
                episodes_completed = batch_size
                total_episodes += episodes_completed
                steps_this_batch_int = int(steps_this_batch.item())
                total_steps += steps_this_batch_int
                
                # Update reward statistics
                p1_total_reward += float(jnp.sum(p1_rewards).item())
                p2_total_reward += float(jnp.sum(p2_rewards).item())
                
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
                    avg_p1_reward = p1_total_reward / total_episodes if total_episodes > 0 else 0
                    avg_p2_reward = p2_total_reward / total_episodes if total_episodes > 0 else 0
                    pbar.set_postfix({
                        "episodes/s": f"{format_human_readable(episodes_per_sec)}",
                        "steps/s": f"{format_human_readable(steps_per_sec)}",
                        f"avg_reward_{self.model1_name}": f"{avg_p1_reward:.3f}",
                        f"avg_reward_{self.model2_name}": f"{avg_p2_reward:.3f}",
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
        
        # Calculate average rewards
        avg_p1_reward = p1_total_reward / total_episodes if total_episodes > 0 else 0
        avg_p2_reward = p2_total_reward / total_episodes if total_episodes > 0 else 0
        
        result = {
            'batch_size': batch_size,
            'steps_per_second': steps_per_second,
            'games_per_second': episodes_per_second,
            'avg_game_length': avg_episode_length,
            'median_game_length': median_episode_length,
            'min_game_length': min_episode_length,
            'max_game_length': max_episode_length,
            'memory_usage_gb': peak_memory_gb,
            'memory_usage_percent': memory_percent,
            'efficiency': efficiency,
            f'avg_reward_{self.model1_name}': avg_p1_reward,
            f'avg_reward_{self.model2_name}': avg_p2_reward,
            'reward_difference': avg_p1_reward - avg_p2_reward,
            'model1_win_rate': len([r for r in episode_lengths if avg_p1_reward > avg_p2_reward]) / len(episode_lengths) if episode_lengths else 0
        }
        
        return result
    
    def print_single_result(self, result: Dict[str, Any], extra_info: Optional[Dict] = None):
        """Print a single benchmark result with model comparison metrics."""
        print(f"\n=== {self.name} Single Batch Result ===")
        print(f"Batch Size: {result['batch_size']}")
        print(f"Steps/Second: {result['steps_per_second']:.2f}")
        print(f"Games/Second: {result['games_per_second']:.2f}")
        print(f"Memory Usage: {result['memory_usage_gb']:.2f} GB ({result['memory_usage_percent']:.1f}%)")
        print(f"Efficiency: {result['efficiency']:.2f} steps/s/GB")
        print(f"Average {self.model1_name} Reward: {result[f'avg_reward_{self.model1_name}']:.3f}")
        print(f"Average {self.model2_name} Reward: {result[f'avg_reward_{self.model2_name}']:.3f}")
        print(f"Reward Difference ({self.model1_name} - {self.model2_name}): {result['reward_difference']:.3f}")
        print(f"{self.model1_name} Win Rate: {result['model1_win_rate']:.3f}")
        
        if extra_info:
            for key, value in extra_info.items():
                print(f"{key}: {value}")
    
    def _run_discovery(self, 
                      memory_limit_gb: float, 
                      duration: int, 
                      custom_batch_sizes: Optional[List[int]],
                      verbose: bool) -> List[Dict[str, Any]]:
        """Run the discovery process for model comparison."""
        results = []
        
        # Use custom batch sizes or generate sequence (avoid batch size 1)
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            batch_sizes = [2, 4, 8, 16, 32]  # Start from batch size 2, avoid issues with batch size 1
        
        # Run benchmarks
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size(batch_size, duration)
                results.append(result)
                
                # Print results
                self.print_single_result(result)
                
                # Check termination conditions  
                if result['memory_usage_percent'] >= 90:
                    print(f"Memory limit reached: {result['memory_usage_percent']:.1f}%", flush=True)
                    break
                    
                if len(results) > 1:
                    perf_improvement = (results[-1]['games_per_second'] / 
                                      results[-2]['games_per_second'] - 1.0)
                    if perf_improvement < 0.05:
                        print("Diminishing returns detected (less than 5% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        # Generate model comparison graphs
        if len(results) > 1:
            try:
                perf_plot, reward_plot = self._generate_model_comparison_plots(results)
                print(f"\nGenerated comparison plots:")
                print(f"Performance plot: {perf_plot}")
                print(f"Reward plot: {reward_plot}")
            except Exception as e:
                print(f"Error generating plots: {e}")
        
        return results
    
    def _generate_model_comparison_plots(self, results: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Generate model comparison plots showing games/s and average reward per batch size."""
        import matplotlib.pyplot as plt
        from datetime import datetime
        from pathlib import Path
        
        # Create results directory
        results_dir = Path("benchmarks/graphs")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract data for plotting
        batch_sizes = [result['batch_size'] for result in results]
        games_per_second = [result['games_per_second'] for result in results]
        steps_per_second = [result['steps_per_second'] for result in results]
        model1_rewards = [result[f'avg_reward_{self.model1_name}'] for result in results]
        model2_rewards = [result[f'avg_reward_{self.model2_name}'] for result in results]
        reward_differences = [result['reward_difference'] for result in results]
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot 1: Performance - Games/s and Steps/s vs Batch Size
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Primary axis: Games per second
        color1 = 'tab:blue'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Games per Second', color=color1)
        ax1.plot(batch_sizes, games_per_second, 'o-', color=color1, linewidth=2, markersize=8, label='Games/s')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Model Comparison Performance: {self.model1_name} vs {self.model2_name}\\n'
                     f'MCTS: {self.num_simulations} simulations, {self.max_nodes} max nodes')
        
        # Secondary axis: Steps per second
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Steps per Second', color=color2)
        ax2.plot(batch_sizes, steps_per_second, '^-', color=color2, linewidth=2, markersize=8, label='Steps/s')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save performance plot
        performance_plot_path = results_dir / f"model_comparison_{self.model1_name}_vs_{self.model2_name}_{timestamp}_performance.png"
        plt.savefig(performance_plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Rewards - Average Reward per Model vs Batch Size
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Primary axis: Average rewards
        color1 = 'tab:red'
        color2 = 'tab:orange'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Average Reward')
        ax1.plot(batch_sizes, model1_rewards, 'o-', color=color1, linewidth=2, markersize=8, 
                label=f'{self.model1_name} Avg Reward')
        ax1.plot(batch_sizes, model2_rewards, 's-', color=color2, linewidth=2, markersize=8, 
                label=f'{self.model2_name} Avg Reward')
        ax1.set_xscale('log', base=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Model Comparison Rewards: {self.model1_name} vs {self.model2_name}\\n'
                     f'MCTS: {self.num_simulations} simulations, {self.max_nodes} max nodes')
        
        # Secondary axis: Reward difference
        ax2 = ax1.twinx()
        color3 = 'tab:purple'
        ax2.set_ylabel('Reward Difference (Model1 - Model2)', color=color3)
        ax2.plot(batch_sizes, reward_differences, 'D--', color=color3, linewidth=2, markersize=6, 
                alpha=0.7, label='Reward Difference')
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)  # Zero line
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save reward plot
        reward_plot_path = results_dir / f"model_comparison_{self.model1_name}_vs_{self.model2_name}_{timestamp}_rewards.png"
        plt.savefig(reward_plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        return str(performance_plot_path), str(reward_plot_path)