"""Game worker for distributed self-play game generation.

This worker runs batches of self-play games using StochasticMCTS,
generating training experiences and sending them to the Redis replay buffer.
"""

import os
import time
from functools import partial
from typing import Dict, Any, Optional, Callable, Tuple

import jax
import jax.numpy as jnp
import chex
import ray

from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.common import step_env_and_evaluator
from core.types import StepMetadata
from core.memory.replay_memory import BaseExperience

from .base_worker import BaseWorker, WorkerStats
from ..serialization import (
    serialize_experience,
    serialize_rewards,
    deserialize_weights,
)
from ..buffer.redis_buffer import RedisReplayBuffer


@ray.remote
class GameWorker(BaseWorker):
    """Ray actor that generates self-play games using StochasticMCTS.

    Each worker runs a batch of parallel environments, collecting experiences
    and sending them to the centralized Redis replay buffer.

    Example:
        >>> coordinator = create_coordinator(config)
        >>> worker = GameWorker.remote(coordinator, config={'batch_size': 32})
        >>> ray.get(worker.run.remote(num_iterations=1000))
    """

    def __init__(
        self,
        coordinator_handle: ray.actor.ActorHandle,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the game worker.

        Args:
            coordinator_handle: Ray actor handle for the coordinator.
            worker_id: Optional unique worker ID. Auto-generated if not provided.
            config: Configuration dict with keys:
                - batch_size: Number of parallel environments (default: 16)
                - num_simulations: MCTS simulations per move (default: 100)
                - max_nodes: Maximum MCTS tree nodes (default: 400)
                - max_episode_steps: Max steps per episode (default: 500)
                - temperature: MCTS temperature (default: 1.0)
                - redis_host: Redis server host (default: 'localhost')
                - redis_port: Redis server port (default: 6379)
        """
        super().__init__(coordinator_handle, worker_id, config)

        # Log backend info
        print(f"GameWorker: JAX backend = {jax.default_backend()}")

        # Game configuration
        self.batch_size = self.config.get('batch_size', 16)
        self.num_simulations = self.config.get('num_simulations', 100)
        self.max_nodes = self.config.get('max_nodes', 400)
        self.max_episode_steps = self.config.get('max_episode_steps', 500)
        self.temperature = self.config.get('temperature', 1.0)

        # Initialize Redis buffer
        redis_host = self.config.get('redis_host', 'localhost')
        redis_port = self.config.get('redis_port', 6379)
        redis_password = self.config.get('redis_password', None)
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
                state.is_stochastic,
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
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6

            @nn.compact
            def __call__(self, x, train: bool = False):
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                value = nn.Dense(1)(x)
                value = jnp.squeeze(value, axis=-1)
                return policy_logits, value

        self._nn_model = ResNetTurboZero(self._env.num_actions, num_hidden=256, num_blocks=6)

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
        """Initialize neural network parameters from scratch or from coordinator."""
        # Try to get weights from coordinator
        weights_bytes = self.fetch_model_weights()

        if weights_bytes is not None:
            # Use coordinator weights
            params_dict = deserialize_weights(weights_bytes)
            self._nn_params = params_dict
            print(f"Worker {self.worker_id}: Loaded model version {self.current_model_version}")
        else:
            # Initialize random weights
            key = jax.random.PRNGKey(42)
            sample_state, _ = self._env_init_fn(key)
            sample_obs = self._state_to_nn_input_fn(sample_state)
            variables = self._nn_model.init(key, sample_obs[None, ...], train=False)
            self._nn_params = {'params': variables['params']}
            print(f"Worker {self.worker_id}: Initialized random weights")

    def _init_collection_state(self) -> Dict[str, Any]:
        """Initialize the collection state for batched game playing."""
        key, self._rng_key = jax.random.split(self._rng_key)
        init_keys = jax.random.split(key, self.batch_size)

        # Initialize environments
        env_states, metadatas = jax.vmap(self._env_init_fn)(init_keys)

        # Initialize evaluator states
        template_state, _ = self._env_init_fn(jax.random.PRNGKey(0))
        eval_states = self._evaluator.init_batched(
            self.batch_size,
            template_embedding=template_state
        )

        return {
            'env_states': env_states,
            'metadatas': metadatas,
            'eval_states': eval_states,
            'episode_experiences': [[] for _ in range(self.batch_size)],
            'episode_ids': [None] * self.batch_size,
        }

    def _on_model_update_available(self, new_version: int) -> None:
        """Handle new model version notification."""
        weights_bytes = self.fetch_model_weights()
        if weights_bytes is not None:
            self._nn_params = deserialize_weights(weights_bytes)
            print(f"Worker {self.worker_id}: Updated to model version {new_version}")

    def _collect_step(self) -> int:
        """Run one step of game collection across all batch environments.

        Returns:
            Number of completed episodes in this step.
        """
        key, self._rng_key = jax.random.split(self._rng_key)

        state = self._collection_state
        env_states = state['env_states']
        metadatas = state['metadatas']
        eval_states = state['eval_states']

        # Create step function
        step_fn = partial(
            step_env_and_evaluator,
            evaluator=self._evaluator,
            env_step_fn=self._env_step_fn,
            env_init_fn=self._env_init_fn,
            max_steps=self.max_episode_steps
        )

        # Step all environments
        step_keys = jax.random.split(key, self.batch_size)

        # Vectorize the step function
        def batched_step(env_state, metadata, eval_state, params, key):
            return step_fn(
                key=key,
                env_state=env_state,
                env_state_metadata=metadata,
                eval_state=eval_state,
                params=params
            )

        vmapped_step = jax.vmap(batched_step, in_axes=(0, 0, 0, None, 0))

        eval_outputs, new_env_states, new_metadatas, terminateds, truncateds, rewards_batch = vmapped_step(
            env_states,
            metadatas,
            eval_states,
            self._nn_params,
            step_keys
        )

        # Process experiences and handle episode completion
        completed_episodes = 0

        for i in range(self.batch_size):
            # Skip stochastic states (dice rolls)
            is_stochastic = env_states.is_stochastic[i] if hasattr(env_states, 'is_stochastic') else False

            if not is_stochastic:
                # Collect experience for training
                exp = {
                    'observation_nn': self._state_to_nn_input_fn(
                        jax.tree.map(lambda x: x[i], env_states)
                    ),
                    'policy_weights': eval_outputs.policy_weights[i],
                    'policy_mask': metadatas.action_mask[i],
                    'cur_player_id': metadatas.cur_player_id[i],
                }
                state['episode_experiences'][i].append(exp)

            terminated = terminateds[i]
            truncated = truncateds[i]

            if terminated or truncated:
                # Episode complete - send to Redis
                if terminated and len(state['episode_experiences'][i]) > 0:
                    self._send_episode_to_buffer(
                        state['episode_experiences'][i],
                        rewards_batch[i],
                    )
                    completed_episodes += 1
                    self.stats.games_generated += 1

                # Reset episode state
                state['episode_experiences'][i] = []
                state['episode_ids'][i] = None

        # Update collection state
        self._collection_state['env_states'] = new_env_states
        self._collection_state['metadatas'] = new_metadatas
        self._collection_state['eval_states'] = eval_outputs.eval_state

        return completed_episodes

    def _send_episode_to_buffer(
        self,
        experiences: list,
        final_rewards: jnp.ndarray,
    ) -> None:
        """Send a completed episode to the Redis buffer.

        Args:
            experiences: List of experience dicts from the episode.
            final_rewards: Final rewards for each player.
        """
        # Serialize experiences
        serialized_exps = []
        for exp in experiences:
            exp_bytes = serialize_experience(exp)
            serialized_exps.append(exp_bytes)

        # Serialize rewards
        rewards_bytes = serialize_rewards(final_rewards)

        # Add to Redis buffer
        try:
            self.buffer.add_episode(
                experiences=serialized_exps,
                final_rewards=rewards_bytes,
                model_version=self.current_model_version,
                metadata={
                    'worker_id': self.worker_id,
                    'episode_length': len(experiences),
                }
            )
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

        print(f"Worker {self.worker_id}: Starting game collection loop...")

        iteration = 0
        total_games = 0
        total_steps = 0
        start_time = time.time()

        while self.running:
            if num_iterations >= 0 and iteration >= num_iterations:
                break

            # Collect one step across all environments
            completed = self._collect_step()
            total_games += completed
            total_steps += self.batch_size
            iteration += 1

            # Log progress periodically
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = total_steps / max(elapsed, 0.001)
                games_per_min = (total_games / max(elapsed, 0.001)) * 60
                print(
                    f"Worker {self.worker_id}: "
                    f"iter={iteration}, games={total_games}, "
                    f"steps/s={steps_per_sec:.1f}, games/min={games_per_min:.1f}"
                )

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
        """Get current collection statistics.

        Returns:
            Dict with collection stats.
        """
        base_stats = self.get_stats()
        base_stats.update({
            'batch_size': self.batch_size,
            'num_simulations': self.num_simulations,
            'max_nodes': self.max_nodes,
            'temperature': self.temperature,
            'buffer_size': self.buffer.get_size() if self.buffer else 0,
        })
        return base_stats
