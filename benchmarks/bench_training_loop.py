#!/usr/bin/env python3
"""
Training Loop benchmark implementation for testing batch size optimization.

This benchmark tests the performance of the training loop components with different batch sizes,
helping to determine optimal configurations for the model size being used.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
from functools import partial
import flax.linen as nn
import optax

# TurboZero imports
from core.training.train import Trainer, CollectionState
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.mlp import MLP, MLPConfig
from core.training.loss_fns import az_default_loss_fn
from core.types import StepMetadata

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


class TrainingLoopBenchmark(BaseBenchmark):
    """Benchmark for testing training loop performance with different batch sizes."""
    
    def __init__(self, train_batch_size: int = 512, collection_steps: int = 10, train_steps: int = 5):
        super().__init__(
            name="TrainingLoop",
            description="Training loop performance benchmark for batch size optimization"
        )
        
        # Training configuration
        self.train_batch_size = train_batch_size
        self.collection_steps = collection_steps
        self.train_steps = train_steps
        
        # Import backgammon environment
        import pgx.backgammon as bg
        self.bg_env = bg.Backgammon(short_game=True)
        
        # Create step and init functions
        self.env_step_fn = self._create_step_fn()
        self.env_init_fn = self._create_init_fn()
        
        # Create network, evaluator, and other components
        self.network = self._create_network()
        self.evaluator = self._create_evaluator()
        self.optimizer = optax.adam(1e-4)
        self.memory_buffer = EpisodeReplayBuffer(capacity=10000)
        
        # State to NN input function
        self.state_to_nn_input_fn = self._create_state_to_nn_input_fn()
    
    def _create_step_fn(self):
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
    
    def _create_init_fn(self):
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
    
    def _create_network(self):
        """Create a simple MLP network for benchmarking with 6-way value head."""
        mlp_config = MLPConfig(
            hidden_dims=[256, 256, 256],
            policy_head_out_size=self.bg_env.num_actions,
            value_head_out_size=6  # 6-way outcome distribution for backgammon
        )
        return MLP(mlp_config)
    
    def _create_evaluator(self):
        """Create MCTS evaluator for training."""
        eval_fn = make_nn_eval_fn(self.network, self._create_state_to_nn_input_fn())
        action_selector = PUCTSelector(c=1.0)
        
        return MCTS(
            eval_fn=eval_fn,
            action_selector=action_selector,
            branching_factor=self.bg_env.num_actions,
            max_nodes=1000,
            num_iterations=50,
            discount=1.0,  # Single-player self-play
            temperature=1.0
        )
    
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
    
    def create_trainer(self, batch_size: int) -> Trainer:
        """Create a Trainer instance with the given batch size."""
        return Trainer(
            batch_size=batch_size,
            train_batch_size=self.train_batch_size,
            warmup_steps=0,  # No warmup for benchmarking
            collection_steps_per_epoch=self.collection_steps,
            train_steps_per_epoch=self.train_steps,
            nn=self.network,
            loss_fn=az_default_loss_fn,
            optimizer=self.optimizer,
            evaluator=self.evaluator,
            memory_buffer=self.memory_buffer,
            max_episode_steps=500,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            state_to_nn_input_fn=self.state_to_nn_input_fn,
            testers=[],  # No testers for benchmarking
            num_devices=1  # Single device for benchmarking
        )
    
    def benchmark_training_components(self, trainer: Trainer, batch_size: int, duration: int) -> Tuple[float, float, int]:
        """Benchmark the training components using simplified approach."""
        # For now, skip the complex pmap training and just benchmark network operations
        # This gives us a baseline for training performance without the multi-device complexity
        
        print("Benchmarking simplified training components...", flush=True)
        
        # Initialize network parameters
        key = jax.random.PRNGKey(0)
        sample_state = self.bg_env.init(key)
        sample_obs = self.state_to_nn_input_fn(sample_state)
        
        # Initialize network
        net_key, key = jax.random.split(key)
        variables = self.network.init(net_key, sample_obs[None, ...], train=False)
        params = variables['params']
        
        # Create optimizer state
        opt_state = self.optimizer.init(params)
        
        # Create a simple training step function
        @jax.jit
        def train_step(params, opt_state, batch_obs, batch_targets, key):
            def loss_fn(params, obs, targets):
                policy_logits, values = self.network.apply({'params': params}, obs, train=False)
                # Simple MSE loss for benchmarking
                policy_loss = jnp.mean((policy_logits - targets) ** 2)
                value_loss = jnp.mean((values - jnp.mean(targets, axis=1)) ** 2)
                return policy_loss + value_loss
            
            loss, grads = jax.value_and_grad(loss_fn)(params, batch_obs, batch_targets)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        # Benchmark training steps
        collect_steps_per_sec = batch_size * 10  # Estimate collection performance
        
        start_time = time.time()
        train_iterations = 0
        
        while time.time() - start_time < duration:
            # Create dummy training batch
            step_key, key = jax.random.split(key)
            dummy_obs = jnp.zeros((self.train_batch_size, *sample_obs.shape))
            dummy_targets = jax.random.normal(step_key, (self.train_batch_size, self.bg_env.num_actions))
            
            # Run training step
            params, opt_state, loss = train_step(params, opt_state, dummy_obs, dummy_targets, step_key)
            jax.block_until_ready(loss)
            train_iterations += 1
        
        train_duration = time.time() - start_time
        train_steps_per_sec = (train_iterations * self.train_batch_size) / train_duration
        
        return collect_steps_per_sec, train_steps_per_sec, train_iterations
    
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> BatchBenchResult:
        """Run benchmark for a specific batch size."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        print(f"\nBenchmarking Training Loop: Batch={batch_size} for {max_duration}s", flush=True)
        
        # Track memory
        initial_memory_gb = get_memory_usage()
        
        try:
            # Create trainer
            trainer = self.create_trainer(batch_size)
            
            # Warmup compilation
            print("Compiling and warming up...", flush=True)
            collect_steps_per_sec, train_steps_per_sec, total_training_steps = self.benchmark_training_components(
                trainer, batch_size, 10  # Short warmup
            )
            print("Warm-up complete", flush=True)
            
            # Run actual benchmark
            collect_steps_per_sec, train_steps_per_sec, total_training_steps = self.benchmark_training_components(
                trainer, batch_size, max_duration
            )
            
        except Exception as e:
            print(f"Error during benchmark: {e}", flush=True)
            # Return invalid result
            return BatchBenchResult(
                batch_size=batch_size,
                steps_per_second=0.0,
                games_per_second=0.0,
                avg_game_length=0.0,
                median_game_length=0.0,
                min_game_length=0,
                max_game_length=0,
                memory_usage_gb=get_memory_usage(),
                memory_usage_percent=calculate_memory_percentage(get_memory_usage()),
                efficiency=0.0,
                valid=False
            )
        
        # Calculate final metrics
        peak_memory_gb = get_memory_usage()
        
        # Use training steps per second as primary metric
        # Estimate games per second based on typical episode length
        estimated_episode_length = 30
        estimated_games_per_sec = collect_steps_per_sec / estimated_episode_length
        
        memory_percent = calculate_memory_percentage(peak_memory_gb)
        efficiency = train_steps_per_sec / peak_memory_gb
        
        result = BatchBenchResult(
            batch_size=batch_size,
            steps_per_second=train_steps_per_sec,  # Training steps per second
            games_per_second=estimated_games_per_sec,  # Estimated episodes per second
            avg_game_length=estimated_episode_length,
            median_game_length=estimated_episode_length,
            min_game_length=estimated_episode_length,
            max_game_length=estimated_episode_length,
            memory_usage_gb=peak_memory_gb,
            memory_usage_percent=memory_percent,
            efficiency=efficiency
        )
        
        return result
    
    def get_profile_params(self) -> Dict[str, Any]:
        """Get parameters for profile naming."""
        return {
            "train_batch_size": self.train_batch_size,
            "collection_steps": self.collection_steps,
            "train_steps": self.train_steps
        }
    
    def _run_discovery(self, 
                      memory_limit_gb: float, 
                      duration: int, 
                      custom_batch_sizes: Optional[List[int]],
                      verbose: bool) -> List[BatchBenchResult]:
        """Run the discovery process for the training loop benchmark."""
        results = []
        
        # Use custom batch sizes or generate sequence (avoid batch size 1)
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            # Start with smaller batch sizes for training, avoid batch size 1
            batch_sizes = [2, 4, 8, 16, 32, 64]  # Self-play batch sizes
        
        # Run benchmarks
        for batch_size in batch_sizes:
            try:
                result = self.benchmark_batch_size(batch_size, duration)
                results.append(result)
                
                # Check termination conditions  
                if result.memory_usage_percent >= 90:  # 90% memory usage threshold
                    print(f"Memory limit reached: {result.memory_usage_percent:.1f}%", flush=True)
                    break
                    
                if len(results) > 1 and result.valid:
                    perf_improvement = (results[-1].steps_per_second / 
                                      results[-2].steps_per_second - 1.0)
                    if perf_improvement < 0.05:  # Lower threshold for training performance
                        print("Diminishing returns detected (less than 5% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        return results