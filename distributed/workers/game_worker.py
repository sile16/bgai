"""Game worker for distributed self-play game generation.

This worker runs batches of self-play games using StochasticMCTS,
generating training experiences and sending them to the Redis replay buffer.

No Ray dependency - uses Redis for all coordination.
"""

import queue
import threading
import time
from functools import partial
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp

from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.equity import terminal_value_probs_from_reward_4way, probs_to_equity_4way
from core.common import step_env_and_evaluator
from core.types import StepMetadata

from .base_worker import BaseWorker
from ..serialization import (
    serialize_experience,
    serialize_rewards,
    deserialize_weights,
    deserialize_warm_tree,
)
from ..buffer.redis_buffer import RedisReplayBuffer
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint, WorkerPhase


class GameWorker(BaseWorker):
    """Worker that generates self-play games using StochasticMCTS.

    Each worker runs a batch of parallel environments, collecting experiences
    and sending them to the centralized Redis replay buffer.

    Example:
        >>> worker = GameWorker(config={'batch_size': 32})
        >>> worker.run()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
    ):
        """Initialize the game worker.

        Args:
            config: Configuration dict with keys:
                - batch_size: Number of parallel environments (default: 16)
                - num_simulations: MCTS simulations per move (default: 100)
                - max_nodes: Maximum MCTS tree nodes (default: 400)
                - max_episode_steps: Max steps per episode (default: 500)
                - temperature: MCTS temperature (default: 1.0)
                - redis_host: Redis server host (default: 'localhost')
                - redis_port: Redis server port (default: 6379)
                - redis_password: Redis password (optional)
            worker_id: Optional unique worker ID. Auto-generated if not provided.
        """
        # Initialize base worker (handles Redis connection, registration)
        super().__init__(config, worker_id)

        # Log backend info
        print(f"GameWorker: JAX backend = {jax.default_backend()}")

        # Game configuration
        self.batch_size = self.config.get('batch_size', 16)
        self.num_simulations = self.config.get('num_simulations', 100)
        self.max_nodes = self.config.get('max_nodes', 400)
        self.max_episode_steps = self.config.get('max_episode_steps', 500)

        # Temperature schedule is now managed by the Coordinator via Redis.
        # We just fetch the current temperature when updating the model.
        self.temperature = self.config.get('temperature', 1.0)
        self._current_temperature = self.temperature

        # Initialize Redis buffer (uses same connection info)
        redis_host = self.config.get('redis_host', 'localhost')
        redis_port = self.config.get('redis_port', 6379)
        redis_password = self.config.get('redis_password')
        self.buffer = RedisReplayBuffer(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            worker_id=self.worker_id,
        )

        # Environment setup (lazy initialization)
        self._env = None
        self._env_step_fn = None
        self._env_init_fn = None
        self._state_to_nn_input_fn = None

        # Evaluator (lazy initialization)
        self._evaluator = None
        self._nn_model = None
        self._nn_params = None

        # Collection state
        self._collection_state = None
        self._rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

        # Timing accumulators for GPU vs CPU breakdown
        self._gpu_time_accum = 0.0
        self._cpu_time_accum = 0.0
        # Detailed CPU timing breakdown
        self._cpu_loop_time = 0.0
        self._cpu_exp_build_time = 0.0
        self._cpu_value_pred_time = 0.0
        self._cpu_send_episode_time = 0.0

        # Warm tree state
        self._warm_tree = None
        self._warm_tree_version = 0

        # Async Redis sender - queue completed episodes for background sending
        # This allows GPU to continue with next batch while Redis I/O happens in parallel
        self._send_queue = queue.Queue(maxsize=200)  # Buffer up to 200 episodes
        self._send_thread = None  # Started lazily in _run_loop
        self._send_thread_running = False
        self._episodes_queued = 0
        self._episodes_sent = 0

    def _start_async_sender(self):
        """Start the background thread for async Redis sending."""
        if self._send_thread is not None:
            return  # Already started
        self._send_thread_running = True
        self._send_thread = threading.Thread(
            target=self._async_send_worker,
            name=f"redis-sender-{self.worker_id}",
            daemon=True,
        )
        self._send_thread.start()

    def _stop_async_sender(self):
        """Stop the background sender thread and flush remaining episodes."""
        if self._send_thread is None:
            return
        self._send_thread_running = False
        # Signal thread to stop by putting None
        try:
            self._send_queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        # Wait for thread to finish (with timeout)
        self._send_thread.join(timeout=5.0)
        if self._send_thread.is_alive():
            print(f"Worker {self.worker_id}: Warning - async sender thread did not stop cleanly")

    def _async_send_worker(self):
        """Background thread that sends episodes to Redis.

        Runs continuously, pulling episodes from the queue and sending them
        to Redis. This allows the main collection loop to continue without
        blocking on Redis I/O.
        """
        while self._send_thread_running or not self._send_queue.empty():
            try:
                episode_data = self._send_queue.get(timeout=1.0)
                if episode_data is None:  # Shutdown signal
                    break
                # Send to Redis
                self.buffer.add_episode(**episode_data)
                self._episodes_sent += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id}: Async send error: {e}")
        # Drain any remaining items on shutdown
        while not self._send_queue.empty():
            try:
                episode_data = self._send_queue.get_nowait()
                if episode_data is not None:
                    self.buffer.add_episode(**episode_data)
                    self._episodes_sent += 1
            except queue.Empty:
                break
            except Exception as e:
                print(f"Worker {self.worker_id}: Async send drain error: {e}")

    @property
    def worker_type(self) -> str:
        return 'game'

    def _setup_environment(self) -> None:
        """Set up the backgammon environment."""
        import pgx.backgammon as bg

        self._env = bg.Backgammon(short_game=True)

        # Create step function that handles stochastic states
        def step_fn(state, action, key):
            def stochastic_branch(operand):
                s, a, _ = operand
                return self._env.stochastic_step(s, a)

            def deterministic_branch(operand):
                s, a, k = operand
                return self._env.step(s, a, k)

            new_state = jax.lax.cond(
                state._is_stochastic,
                stochastic_branch,
                deterministic_branch,
                (state, action, key)
            )

            metadata = StepMetadata(
                rewards=new_state.rewards,
                action_mask=new_state.legal_action_mask,
                terminated=new_state.terminated,
                cur_player_id=new_state.current_player,
                step=new_state._step_count
            )

            return new_state, metadata

        self._env_step_fn = step_fn

        def init_fn(key):
            state = self._env.init(key)
            return state, StepMetadata(
                rewards=state.rewards,
                action_mask=state.legal_action_mask,
                terminated=state.terminated,
                cur_player_id=state.current_player,
                step=state._step_count
            )

        self._env_init_fn = init_fn
        self._state_to_nn_input_fn = lambda state: state.observation

    def _setup_neural_network(self) -> None:
        """Set up the neural network model."""
        import flax.linen as nn

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
            """ResNet-style network with 4-way conditional value head for backgammon.

            Value head outputs 4 independent sigmoid logits:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
            """
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6
            value_head_out_size: int = 4  # 4-way conditional probabilities

            @nn.compact
            def __call__(self, x, train: bool = False):  # noqa: ARG002 - train required by interface
                del train  # unused but required by turbozero interface
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                # 4-way value head: outputs sigmoid logits
                value_logits = nn.Dense(self.value_head_out_size)(x)
                return policy_logits, value_logits

        # Use network config from YAML, with fallback to defaults
        num_hidden = self.config.get('network_hidden_dim', 256)
        num_blocks = self.config.get('network_num_blocks', 6)
        self._nn_model = ResNetTurboZero(self._env.num_actions, num_hidden=num_hidden, num_blocks=num_blocks)

    def _setup_evaluator(self) -> None:
        """Set up the MCTS evaluator."""
        if self._nn_model is None:
            self._setup_neural_network()

        eval_fn = make_nn_eval_fn(self._nn_model, self._state_to_nn_input_fn)

        self._evaluator = StochasticMCTS(
            eval_fn=eval_fn,
            stochastic_action_probs=self._env.stochastic_action_probs,
            num_iterations=self.num_simulations,
            max_nodes=self.max_nodes,
            branching_factor=self._env.num_actions,
            action_selector=PUCTSelector(),
            temperature=self.temperature,
        )

    def _initialize_params(self) -> None:
        """Initialize neural network parameters from scratch or from Redis."""
        # Try to get weights from Redis
        result = self.get_current_model_weights()

        if result is not None:
            weights_bytes, version = result
            params_dict = deserialize_weights(weights_bytes)
            self._nn_params = params_dict
            print(f"Worker {self.worker_id}: Loaded model version {version}")

            # Also try to get warm tree
            self._check_and_update_warm_tree()
        else:
            # Initialize random weights
            key = jax.random.PRNGKey(42)
            sample_state, _ = self._env_init_fn(key)
            sample_obs = self._state_to_nn_input_fn(sample_state)
            variables = self._nn_model.init(key, sample_obs[None, ...])
            self._nn_params = {'params': variables['params']}
            print(f"Worker {self.worker_id}: Initialized random weights")

    def _init_collection_state(self) -> Dict[str, Any]:
        """Initialize the collection state for batched game playing."""
        key, self._rng_key = jax.random.split(self._rng_key)
        init_keys = jax.random.split(key, self.batch_size)

        # Initialize environments
        env_states, metadatas = jax.vmap(self._env_init_fn)(init_keys)

        # Initialize evaluator states - use warm tree if available
        template_state, _ = self._env_init_fn(jax.random.PRNGKey(0))

        if self._warm_tree is not None:
            # Replicate warm tree across batch dimension
            try:
                eval_states = self._replicate_warm_tree(self._warm_tree)
                print(f"Worker {self.worker_id}: Using warm tree for game initialization")
            except Exception as e:
                print(f"Worker {self.worker_id}: Warm tree replication failed ({e}), using empty trees")
                self._warm_tree = None  # Disable warm tree for this worker
                eval_states = self._evaluator.init_batched(
                    self.batch_size,
                    template_embedding=template_state
                )
        else:
            # Create empty trees
            eval_states = self._evaluator.init_batched(
                self.batch_size,
                template_embedding=template_state
            )

        return {
            'env_states': env_states,
            'metadatas': metadatas,
            'eval_states': eval_states,
            'episode_experiences': [[] for _ in range(self.batch_size)],
            'episode_value_preds': [[] for _ in range(self.batch_size)],
            'episode_ids': [None] * self.batch_size,
            'episode_steps': [0] * self.batch_size,  # Track steps per episode for temperature schedule
        }

    def _replicate_warm_tree(self, warm_tree):
        """Replicate a single warm tree across the batch dimension.

        Args:
            warm_tree: A single MCTSTree to replicate.

        Returns:
            Batched MCTSTree with the warm tree replicated batch_size times.
        """
        # Stack the warm tree batch_size times along a new batch dimension
        def stack_leaves(x):
            return jnp.stack([x] * self.batch_size, axis=0)

        return jax.tree_util.tree_map(stack_leaves, warm_tree)

    def _check_and_update_model(self) -> None:
        """Check for and apply model updates from Redis.
        
        Also updates MCTS temperature from Redis if changed (only when model is updated).
        """
        result = self.get_model_weights()
        if result is not None:
            weights_bytes, version = result
            self._nn_params = deserialize_weights(weights_bytes)
            print(f"Worker {self.worker_id}: Updated to model version {version}")

            # Update Temperature (only when model weights are updated)
            try:
                new_temp = self.state.get_model_temperature()
                if abs(new_temp - self._current_temperature) > 1e-4:
                    print(f"Worker {self.worker_id}: Updating temperature {self._current_temperature:.2f} -> {new_temp:.2f} due to model update")
                    self._current_temperature = new_temp
                    self._evaluator.temperature = new_temp
                    # Re-JIT the step function with new temperature constant
                    self._setup_batched_step()
            except Exception as e:
                print(f"Worker {self.worker_id}: Error updating temperature: {e}")

            # Also check for warm tree update
            self._check_and_update_warm_tree()

    def _check_and_update_warm_tree(self) -> None:
        """Check for and fetch warm tree update from Redis."""
        result = self.state.get_warm_tree_if_newer(self._warm_tree_version)
        if result is not None:
            tree_bytes, version = result
            try:
                self._warm_tree = deserialize_warm_tree(tree_bytes)
                self._warm_tree_version = version
                print(f"Worker {self.worker_id}: Loaded warm tree version {version}")
            except Exception as e:
                print(f"Worker {self.worker_id}: Error loading warm tree: {e}")
                self._warm_tree = None

    def _setup_batched_step(self) -> None:
        """Create and cache the JIT-compiled batched step function."""
        step_fn = partial(
            step_env_and_evaluator,
            evaluator=self._evaluator,
            env_step_fn=self._env_step_fn,
            env_init_fn=self._env_init_fn,
            max_steps=self.max_episode_steps
        )

        def batched_step(env_state, metadata, eval_state, params, key):
            return step_fn(
                key=key,
                env_state=env_state,
                env_state_metadata=metadata,
                eval_state=eval_state,
                params=params
            )

        self._jitted_batched_step = jax.jit(jax.vmap(batched_step, in_axes=(0, 0, 0, None, 0)))

    def _start_gpu_step(self, key):
        """Start GPU computation for one step (non-blocking).

        Returns the GPU results and the input state needed for CPU processing.
        The results are JAX arrays that haven't been materialized yet.
        """
        state = self._collection_state
        env_states = state['env_states']
        metadatas = state['metadatas']
        eval_states = state['eval_states']

        # Step all environments
        step_keys = jax.random.split(key, self.batch_size)

        # Time the GPU step dispatch (non-blocking)
        gpu_start = time.perf_counter()
        eval_outputs, new_env_states, new_metadatas, terminateds, truncateds, rewards_batch = self._jitted_batched_step(
            env_states,
            metadatas,
            eval_states,
            self._nn_params,
            step_keys
        )
        self._gpu_time_accum += time.perf_counter() - gpu_start

        # Return all state needed for CPU processing
        return {
            'gpu_results': (eval_outputs, new_env_states, new_metadatas, terminateds, truncateds, rewards_batch),
            'input_state': (env_states, metadatas, eval_states),
        }

    def _process_step_results(self, step_data) -> int:
        """Process GPU results on CPU (blocking - materializes JAX arrays).

        Returns:
            Number of completed episodes in this step.
        """
        cpu_start = time.perf_counter()

        eval_outputs, new_env_states, new_metadatas, terminateds, truncateds, rewards_batch = step_data['gpu_results']
        env_states, metadatas, eval_states = step_data['input_state']
        state = self._collection_state

        # Process experiences and handle episode completion
        completed_episodes = 0

        # OPTIMIZATION: Batch extract all data from GPU in one go instead of 128 individual extractions
        # This reduces GPU->CPU transfers from 256 tree_map calls to a few batched array operations
        exp_start = time.perf_counter()

        # Extract all observations at once (batched) - state_to_nn_input just returns state.observation
        all_observations = env_states.observation  # Shape: (batch_size, obs_dim)

        # Extract stochastic flags for filtering
        is_stochastic_batch = env_states._is_stochastic if hasattr(env_states, '_is_stochastic') else jnp.zeros(self.batch_size, dtype=bool)

        # Extract all policy data at once (these are already batched from eval_outputs)
        all_policy_weights = eval_outputs.policy_weights  # Shape: (batch_size, num_actions)
        all_action_masks = metadatas.action_mask  # Shape: (batch_size, num_actions)
        all_cur_player_ids = metadatas.cur_player_id  # Shape: (batch_size,)

        self._cpu_exp_build_time += time.perf_counter() - exp_start

        # OPTIMIZATION: Extract all value predictions at once using direct batched array access
        # Instead of 128x jax.tree.map + get_value calls, we access the q-values directly
        # eval_states.data.q has shape (batch_size, max_nodes), root is at index 0
        value_start = time.perf_counter()
        ROOT_INDEX = 0  # MCTSTree.ROOT_INDEX is always 0
        all_value_preds = eval_states.data.q[:, ROOT_INDEX]  # Shape: (batch_size,)
        self._cpu_value_pred_time += time.perf_counter() - value_start

        # Now iterate through environments - all data is already on CPU
        loop_start = time.perf_counter()
        for i in range(self.batch_size):
            is_stochastic = bool(is_stochastic_batch[i])

            if not is_stochastic:
                exp = {
                    'observation_nn': all_observations[i],
                    'policy_weights': all_policy_weights[i],
                    'policy_mask': all_action_masks[i],
                    'cur_player_id': all_cur_player_ids[i],
                }
                state['episode_experiences'][i].append(exp)
                state['episode_steps'][i] += 1  # Track episode progress for temperature

                value_pred = float(all_value_preds[i])
                cur_player = int(all_cur_player_ids[i])
                # get_value returns equity in [0, 1] from current player's perspective
                # Convert to player 0's perspective for consistent surprise scoring
                # For player 0: use value directly (P(p0 wins))
                # For player 1: use 1 - value (opponent's win prob = our loss prob)
                value_pred_p0 = value_pred if cur_player == 0 else 1.0 - value_pred
                state['episode_value_preds'][i].append(value_pred_p0)

            terminated = terminateds[i]
            truncated = truncateds[i]

            if terminated or truncated:
                # Track episode length for both terminated and truncated episodes.
                # Episode steps are incremented when an experience is appended.
                try:
                    episode_len = len(state['episode_experiences'][i])
                    if episode_len > 0:
                        metrics.episode_length.labels(worker_id=self.worker_id).observe(episode_len)
                except Exception:
                    pass

                if truncated and len(state['episode_experiences'][i]) > 0:
                    # Truncated episodes are discarded (no terminal outcome),
                    # but we track them so max_episode_steps issues are visible.
                    metrics.games_truncated_total.labels(
                        worker_id=self.worker_id,
                        worker_type='game',
                        reason='max_episode_steps',
                    ).inc()

                if terminated and not truncated and len(state['episode_experiences'][i]) > 0:
                    send_start = time.perf_counter()
                    self._send_episode_to_buffer(
                        state['episode_experiences'][i],
                        rewards_batch[i],
                        state['episode_value_preds'][i],
                    )
                    self._cpu_send_episode_time += time.perf_counter() - send_start
                    completed_episodes += 1
                    self.stats.games_generated += 1

                state['episode_experiences'][i] = []
                state['episode_value_preds'][i] = []
                state['episode_ids'][i] = None
                state['episode_steps'][i] = 0  # Reset step counter for new episode

        self._cpu_loop_time += time.perf_counter() - loop_start

        # Update state with new values from GPU
        self._collection_state['env_states'] = new_env_states
        self._collection_state['metadatas'] = new_metadatas
        self._collection_state['eval_states'] = eval_outputs.eval_state

        # Track CPU time
        self._cpu_time_accum += time.perf_counter() - cpu_start

        return completed_episodes

    def _collect_step(self) -> int:
        """Run one step of game collection across all batch environments.

        This is the original synchronous version, kept for compatibility.

        Returns:
            Number of completed episodes in this step.
        """
        key, self._rng_key = jax.random.split(self._rng_key)
        step_data = self._start_gpu_step(key)
        return self._process_step_results(step_data)

    def _send_episode_to_buffer(
        self,
        experiences: list,
        final_rewards: jnp.ndarray,
        value_predictions: list,
    ) -> None:
        """Queue a completed episode for async sending to the Redis buffer.

        Serializes the episode data and queues it for the background sender thread.
        This allows the main collection loop to continue without blocking on Redis I/O.
        """
        serialized_exps = []
        for exp in experiences:
            exp_bytes = serialize_experience(exp)
            serialized_exps.append(exp_bytes)

        rewards_bytes = serialize_rewards(final_rewards)

        # Compute surprise score: difference between predicted equity and actual outcome
        # Value predictions are now equity values in [0, 1] from probs_to_equity
        # final_rewards[0] is the point reward for player 0 (1, 2, or 3 for wins)
        surprise_score = 0.0
        if value_predictions:
            mean_value_pred = sum(value_predictions) / len(value_predictions)
            # Convert final reward to equity for comparison
            actual_reward = float(final_rewards[0])
            actual_probs = terminal_value_probs_from_reward_4way(jnp.array(actual_reward))
            actual_equity = float(probs_to_equity_4way(actual_probs, match_score=None))
            surprise_score = abs(mean_value_pred - actual_equity)

        metrics = get_metrics()
        metrics.surprise_score.labels(worker_id=self.worker_id).observe(surprise_score)

        # Queue for async sending instead of blocking
        episode_data = {
            'experiences': serialized_exps,
            'final_rewards': rewards_bytes,
            'model_version': self.current_model_version,
            'metadata': {
                'worker_id': self.worker_id,
                'episode_length': len(experiences),
                'mean_value_pred': float(sum(value_predictions) / len(value_predictions)) if value_predictions else 0.0,
            },
            'surprise_score': surprise_score,
        }

        try:
            # Use put with timeout to avoid blocking indefinitely if queue is full
            self._send_queue.put(episode_data, timeout=5.0)
            self._episodes_queued += 1
        except queue.Full:
            # Queue is full - this shouldn't happen often with 200 slot buffer
            # Fall back to synchronous send to avoid data loss
            print(f"Worker {self.worker_id}: Send queue full, falling back to sync send")
            try:
                self.buffer.add_episode(**episode_data)
            except Exception as e:
                print(f"Worker {self.worker_id}: Error sending episode to buffer: {e}")

    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main worker loop.

        Args:
            num_iterations: Number of collection steps to run (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Setup
        print(f"Worker {self.worker_id}: Setting up environment and evaluator...")
        self._setup_environment()
        self._setup_neural_network()
        self._setup_evaluator()
        self._initialize_params()

        # Initialize collection state
        print(f"Worker {self.worker_id}: Initializing {self.batch_size} parallel environments...")
        self._collection_state = self._init_collection_state()

        # Setup and JIT compile the batched step function
        print(f"Worker {self.worker_id}: JIT compiling batched step function...")
        self._setup_batched_step()

        # Log temperature
        print(f"Worker {self.worker_id}: Initial temperature: {self._current_temperature:.2f}")

        # Start Prometheus metrics server
        metrics_port_config = self.config.get('metrics_port', 9100)
        metrics_port = start_metrics_server(metrics_port_config)
        if metrics_port is None:
            print(f"Worker {self.worker_id}: Failed to start metrics server")
            metrics_port = metrics_port_config  # Fallback for registration
        metrics = get_metrics()

        # Register metrics endpoint for dynamic discovery (use actual bound port).
        # If Pushgateway is enabled, don't register an unreachable scrape target.
        if not self.config.get('pushgateway_url'):
            try:
                register_metrics_endpoint(
                    self.buffer.redis,
                    worker_id=self.worker_id,
                    worker_type='game',
                    port=metrics_port,
                    ttl_seconds=300,
                )
                print(f"Worker {self.worker_id}: Registered metrics endpoint on port {metrics_port}")
            except Exception as e:
                print(f"Worker {self.worker_id}: Failed to register metrics endpoint: {e}")
        else:
            print(f"Worker {self.worker_id}: Pushgateway enabled; skipping scrape endpoint registration")

        # Set worker info
        metrics.worker_info.labels(worker_id=self.worker_id).info({
            'type': 'game',
            'batch_size': str(self.batch_size),
            'device': 'gpu' if jax.devices()[0].platform == 'gpu' else 'cpu',
        })
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(1)

        # Set configuration metrics (for correlating with memory usage)
        metrics.worker_batch_size.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(self.batch_size)
        metrics.worker_num_simulations.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(self.num_simulations)
        metrics.worker_max_nodes.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(self.max_nodes)

        # Game worker is always in collecting phase when running
        metrics.worker_phase.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(WorkerPhase.COLLECTING)

        # Pre-initialize truncated counter series so Grafana panels don't disappear
        # when there are zero truncations.
        metrics.games_truncated_total.labels(
            worker_id=self.worker_id,
            worker_type='game',
            reason='max_episode_steps',
        ).inc(0)

        print(f"Worker {self.worker_id}: Starting game collection loop...")

        # Start async Redis sender thread
        self._start_async_sender()
        print(f"Worker {self.worker_id}: Async Redis sender started")

        iteration = 0
        total_games = 0
        total_steps = 0
        start_time = time.time()
        last_model_check = start_time
        last_metrics_refresh = start_time
        last_pause_log = 0

        while self.running:
            if num_iterations >= 0 and iteration >= num_iterations:
                break

            now = time.time()
            if (not self.config.get('pushgateway_url')) and (now - last_metrics_refresh >= 60.0):
                try:
                    register_metrics_endpoint(
                        self.buffer.redis,
                        worker_id=self.worker_id,
                        worker_type='game',
                        port=metrics_port,
                        ttl_seconds=300,
                    )
                except Exception:
                    pass
                last_metrics_refresh = now

            # Check if collection is paused (during training epochs).
            # Only pause GPU collectors; CPU collectors can continue generating games while
            # the trainer uses the head GPU.
            if self.state.is_collection_paused() and jax.default_backend() == "gpu":
                if time.time() - last_pause_log > 10:
                    print(f"Worker {self.worker_id}: Collection paused, waiting for training...")
                    last_pause_log = time.time()
                time.sleep(0.5)  # Wait while paused
                continue

            # Check for model updates periodically (every 10 seconds)
            if time.time() - last_model_check > 10:
                self._check_and_update_model()
                last_model_check = time.time()

            # Collect one step across all environments
            completed = self._collect_step()
            total_games += completed
            total_steps += self.batch_size
            iteration += 1

            # Update Prometheus metrics
            if completed > 0:
                metrics.games_total.labels(
                    worker_id=self.worker_id, worker_type='game'
                ).inc(completed)

            metrics.steps_total.labels(worker_id=self.worker_id).inc(self.batch_size)

            # Log progress periodically
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / max(elapsed, 0.001)
                games_per_min = (total_games / max(elapsed, 0.001)) * 60

                metrics.collection_steps_per_second.labels(
                    worker_id=self.worker_id
                ).set(steps_per_sec)
                metrics.games_per_minute.labels(
                    worker_id=self.worker_id
                ).set(games_per_min)
                metrics.model_version.labels(
                    worker_id=self.worker_id, worker_type='game'
                ).set(self.current_model_version)

                # Log current MCTS temperature
                metrics.mcts_temperature.labels(
                    worker_id=self.worker_id
                ).set(self._current_temperature)

                if not self.config.get('pushgateway_url'):
                    try:
                        register_metrics_endpoint(
                            self.buffer.redis,
                            worker_id=self.worker_id,
                            worker_type='game',
                            port=metrics_port,
                            ttl_seconds=300,
                        )
                    except Exception:
                        pass

                temp_str = f", temp={self._current_temperature:.2f}"

                # Calculate GPU/CPU time breakdown
                total_tracked = self._gpu_time_accum + self._cpu_time_accum
                if total_tracked > 0:
                    gpu_pct = (self._gpu_time_accum / total_tracked) * 100
                    cpu_pct = (self._cpu_time_accum / total_tracked) * 100
                    timing_str = f", GPU={gpu_pct:.0f}%/CPU={cpu_pct:.0f}%"

                    # Detailed CPU breakdown (as % of CPU time)
                    if self._cpu_time_accum > 0:
                        exp_pct = (self._cpu_exp_build_time / self._cpu_time_accum) * 100
                        val_pct = (self._cpu_value_pred_time / self._cpu_time_accum) * 100
                        send_pct = (self._cpu_send_episode_time / self._cpu_time_accum) * 100
                        cpu_detail = f" [exp={exp_pct:.0f}%/val={val_pct:.0f}%/send={send_pct:.0f}%]"
                    else:
                        cpu_detail = ""
                else:
                    timing_str = ""
                    cpu_detail = ""

                print(
                    f"Worker {self.worker_id}: "
                    f"iter={iteration}, games={total_games}, "
                    f"steps/s={steps_per_sec:.1f}, games/min={games_per_min:.1f}, "
                    f"model_v{self.current_model_version}{temp_str}{timing_str}{cpu_detail}"
                )

                # Reset timing accumulators for next reporting period
                self._gpu_time_accum = 0.0
                self._cpu_time_accum = 0.0
                self._cpu_loop_time = 0.0
                self._cpu_exp_build_time = 0.0
                self._cpu_value_pred_time = 0.0
                self._cpu_send_episode_time = 0.0

        # Mark worker as stopped
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='game'
        ).set(0)

        # Stop async sender and flush remaining episodes
        print(f"Worker {self.worker_id}: Stopping async sender (queued={self._episodes_queued}, sent={self._episodes_sent})...")
        self._stop_async_sender()
        print(f"Worker {self.worker_id}: Async sender stopped (final sent={self._episodes_sent})")

        # Cleanup
        self.buffer.close()

        return {
            'status': 'completed',
            'total_iterations': iteration,
            'total_games': total_games,
            'total_steps': total_steps,
            'duration_seconds': time.time() - start_time,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        base_stats = self.get_stats()
        base_stats.update({
            'batch_size': self.batch_size,
            'num_simulations': self.num_simulations,
            'max_nodes': self.max_nodes,
            'temperature': self._current_temperature,
            'buffer_size': self.buffer.get_size() if self.buffer else 0,
        })
        return base_stats


def run_game_worker(config: Optional[Dict[str, Any]] = None, worker_id: Optional[str] = None) -> Dict[str, Any]:
    """Run a game worker (convenience function).

    Args:
        config: Worker configuration.
        worker_id: Optional worker ID.

    Returns:
        Result dict from worker.run().
    """
    worker = GameWorker(config=config, worker_id=worker_id)
    return worker.run()
