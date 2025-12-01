"""Evaluation worker for distributed training.

This worker evaluates the current model against various baselines
after each new epoch (model version update). Multiple eval workers
can run in parallel, picking up different evaluation types.

Eval types:
- gnubg: Play against GNU Backgammon
- random: Play against random policy
- self_play: Play current model against itself (baseline)
- checkpoint_N: Play against a specific checkpoint version
"""

import time
from enum import Enum
from functools import partial
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import chex
import ray

from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.evaluator import EvalOutput
from core.common import step_env_and_evaluator
from core.types import StepMetadata

from .base_worker import BaseWorker, WorkerStats
from ..serialization import deserialize_weights
from ..buffer.redis_buffer import RedisReplayBuffer
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint


class EvalType(Enum):
    """Types of evaluation that can be performed."""
    GNUBG = "gnubg"           # Against GNU Backgammon
    RANDOM = "random"         # Against random policy
    SELF_PLAY = "self_play"   # Current model vs itself
    GREEDY = "greedy"         # Against greedy policy (no MCTS)


@dataclass
class EvalResult:
    """Result of an evaluation run."""
    eval_type: str
    model_version: int
    games_played: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_game_length: float
    avg_points_won: float  # Average points won per game
    duration_seconds: float
    timestamp: float


@ray.remote
class EvalWorker(BaseWorker):
    """Ray actor that evaluates models against baselines.

    Eval workers wait for new model versions and then run evaluation
    games against various opponents. Each worker picks up individual
    eval types from a queue, allowing parallel evaluation.

    Example:
        >>> coordinator = create_coordinator(config)
        >>> worker = EvalWorker.remote(coordinator, config={'eval_games': 100})
        >>> ray.get(worker.run.remote())
    """

    # Redis keys for eval coordination
    EVAL_QUEUE = "bgai:eval:queue"
    EVAL_RESULTS = "bgai:eval:results"
    EVAL_IN_PROGRESS = "bgai:eval:in_progress"

    def __init__(
        self,
        coordinator_handle: ray.actor.ActorHandle,
        worker_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the eval worker.

        Args:
            coordinator_handle: Ray actor handle for the coordinator.
            worker_id: Optional unique worker ID. Auto-generated if not provided.
            config: Configuration dict with keys:
                - eval_games: Games per evaluation (default: 100)
                - batch_size: Number of parallel games (default: 16)
                - num_simulations: MCTS simulations per move (default: 200)
                - max_nodes: Maximum MCTS tree nodes (default: 800)
                - eval_types: List of eval types to run (default: all)
                - redis_host/port/password: Redis connection info
        """
        super().__init__(coordinator_handle, worker_id, config)

        # Log backend info
        print(f"EvalWorker: JAX backend = {jax.default_backend()}")

        # Evaluation configuration
        self.eval_games = self.config.get('eval_games', 100)
        self.batch_size = self.config.get('batch_size', 16)
        self.num_simulations = self.config.get('num_simulations', 200)
        self.max_nodes = self.config.get('max_nodes', 800)
        self.eval_interval = self.config.get('eval_interval', 300)  # seconds

        # Which eval types this worker can run
        self.enabled_eval_types = self.config.get('eval_types', [
            EvalType.GNUBG.value,
            EvalType.RANDOM.value,
            EvalType.SELF_PLAY.value,
        ])

        # Initialize Redis buffer for coordination
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

        # Neural network (lazy initialization)
        self._nn_model = None
        self._nn_params = None

        # Evaluators
        self._mcts_evaluator = None
        self._gnubg_evaluator = None
        self._random_evaluator = None

        # Track last evaluated version
        self._last_evaluated_version = 0

        # RNG key
        self._rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

    @property
    def worker_type(self) -> str:
        return 'evaluation'

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

        self._nn_model = ResNetTurboZero(
            self._env.num_actions,
            num_hidden=256,
            num_blocks=6
        )

    def _setup_mcts_evaluator(self) -> None:
        """Set up the MCTS evaluator for the current model."""
        if self._nn_model is None:
            self._setup_neural_network()

        eval_fn = make_nn_eval_fn(self._nn_model, self._state_to_nn_input_fn)

        self._mcts_evaluator = StochasticMCTS(
            eval_fn=eval_fn,
            stochastic_action_probs=self._env.stochastic_action_probs,
            num_iterations=self.num_simulations,
            max_nodes=self.max_nodes,
            branching_factor=self._env.num_actions,
            action_selector=PUCTSelector(),
            temperature=0.1,  # Lower temperature for evaluation (less exploration)
        )

    def _setup_gnubg_evaluator(self) -> None:
        """Set up the GNUBG evaluator."""
        try:
            from bgai.gnubg_evaluator import GnubgEvaluator
            self._gnubg_evaluator = GnubgEvaluator(self._env)
            print(f"Worker {self.worker_id}: GNUBG evaluator initialized")
        except ImportError as e:
            print(f"Worker {self.worker_id}: GNUBG not available: {e}")
            self._gnubg_evaluator = None

    def _setup_random_evaluator(self) -> None:
        """Set up a random action evaluator."""
        # Simple random evaluator that samples uniformly from legal actions

        class RandomEvaluator:
            def __init__(self, env):
                self.env = env
                self._stochastic_probs = jnp.asarray(
                    env.stochastic_action_probs, dtype=jnp.float32
                )
                self._num_stochastic_actions = int(self._stochastic_probs.shape[0])
                self._num_actions = env.num_actions

            def init(self, *args, **kwargs):
                return jnp.array(0, dtype=jnp.int32)

            def reset(self, state):
                return self.init()

            def get_value(self, state):
                return jnp.array(0.0, dtype=jnp.float32)

            def step(self, state, action):
                return state

            def evaluate(self, key, eval_state, env_state, **kwargs):
                is_stochastic = env_state._is_stochastic

                # Stochastic action
                key1, key2 = jax.random.split(key)
                stochastic_action = jax.random.choice(
                    key1, self._num_stochastic_actions, p=self._stochastic_probs
                )

                # Random legal action
                legal_mask = env_state.legal_action_mask
                # Sample uniformly from legal actions
                probs = jnp.where(legal_mask, 1.0, 0.0)
                probs = probs / jnp.maximum(probs.sum(), 1e-8)
                deterministic_action = jax.random.choice(
                    key2, self._num_actions, p=probs
                )

                action = jnp.where(is_stochastic, stochastic_action, deterministic_action)

                # Create policy weights (uniform over legal)
                policy_weights = jnp.where(legal_mask, 0.0, -jnp.inf)

                return EvalOutput(
                    eval_state=eval_state,
                    action=action,
                    policy_weights=policy_weights
                )

        self._random_evaluator = RandomEvaluator(self._env)

    def _initialize_params(self) -> None:
        """Initialize neural network parameters from coordinator."""
        weights_bytes = self.fetch_model_weights()

        if weights_bytes is not None:
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

    def _on_model_update_available(self, new_version: int) -> None:
        """Handle new model version notification."""
        weights_bytes = self.fetch_model_weights()
        if weights_bytes is not None:
            self._nn_params = deserialize_weights(weights_bytes)
            print(f"Worker {self.worker_id}: Updated to model version {new_version}")

    def _queue_eval_tasks(self, model_version: int) -> None:
        """Queue evaluation tasks for a new model version.

        Only queues tasks if they don't already exist.
        """
        for eval_type in self.enabled_eval_types:
            task_id = f"v{model_version}:{eval_type}"

            # Check if already queued or in progress
            if self.buffer.redis.sismember(self.EVAL_IN_PROGRESS, task_id):
                continue

            # Check if result already exists
            result_key = f"{self.EVAL_RESULTS}:{task_id}"
            if self.buffer.redis.exists(result_key):
                continue

            # Queue the task
            self.buffer.redis.rpush(self.EVAL_QUEUE, task_id)
            print(f"Worker {self.worker_id}: Queued eval task {task_id}")

    def _claim_eval_task(self) -> Optional[Tuple[int, str]]:
        """Try to claim an evaluation task from the queue.

        Returns:
            Tuple of (model_version, eval_type) or None if no tasks available.
        """
        # Try to pop a task from the queue
        task_bytes = self.buffer.redis.lpop(self.EVAL_QUEUE)
        if task_bytes is None:
            return None

        task_id = task_bytes.decode() if isinstance(task_bytes, bytes) else task_bytes

        # Mark as in progress
        self.buffer.redis.sadd(self.EVAL_IN_PROGRESS, task_id)

        # Parse task_id: "vN:eval_type"
        try:
            version_str, eval_type = task_id.split(":", 1)
            model_version = int(version_str[1:])  # Remove 'v' prefix
            return (model_version, eval_type)
        except ValueError:
            print(f"Worker {self.worker_id}: Invalid task_id: {task_id}")
            return None

    def _complete_eval_task(self, result: EvalResult) -> None:
        """Mark an evaluation task as complete and store results."""
        task_id = f"v{result.model_version}:{result.eval_type}"

        # Store result
        result_key = f"{self.EVAL_RESULTS}:{task_id}"
        result_data = {
            'eval_type': result.eval_type,
            'model_version': result.model_version,
            'games_played': result.games_played,
            'wins': result.wins,
            'losses': result.losses,
            'draws': result.draws,
            'win_rate': result.win_rate,
            'avg_game_length': result.avg_game_length,
            'avg_points_won': result.avg_points_won,
            'duration_seconds': result.duration_seconds,
            'timestamp': result.timestamp,
            'worker_id': self.worker_id,
        }
        self.buffer.redis.hset(result_key, mapping={
            k: str(v) for k, v in result_data.items()
        })

        # Store in sorted set for easy querying by version
        self.buffer.redis.zadd(
            f"{self.EVAL_RESULTS}:by_version:{result.eval_type}",
            {task_id: result.model_version}
        )

        # Remove from in-progress set
        self.buffer.redis.srem(self.EVAL_IN_PROGRESS, task_id)

        print(
            f"Worker {self.worker_id}: Completed {result.eval_type} v{result.model_version}: "
            f"win_rate={result.win_rate:.2%}"
        )

    def _run_evaluation(
        self,
        model_version: int,
        eval_type: str,
    ) -> Optional[EvalResult]:
        """Run evaluation games against a specific opponent.

        Args:
            model_version: Model version to evaluate.
            eval_type: Type of evaluation to run.

        Returns:
            EvalResult or None if evaluation failed.
        """
        start_time = time.time()

        # Ensure we have the right model version
        if self.current_model_version != model_version:
            weights_bytes = self.fetch_model_weights()
            if weights_bytes is not None:
                self._nn_params = deserialize_weights(weights_bytes)

        # Get opponent evaluator
        if eval_type == EvalType.GNUBG.value:
            if self._gnubg_evaluator is None:
                print(f"Worker {self.worker_id}: GNUBG not available, skipping")
                return None
            opponent_evaluator = self._gnubg_evaluator
        elif eval_type == EvalType.RANDOM.value:
            opponent_evaluator = self._random_evaluator
        elif eval_type == EvalType.SELF_PLAY.value:
            opponent_evaluator = self._mcts_evaluator
        else:
            print(f"Worker {self.worker_id}: Unknown eval type: {eval_type}")
            return None

        # Run evaluation games
        wins, losses, draws = 0, 0, 0
        total_game_length = 0
        total_points = 0.0

        games_per_batch = min(self.batch_size, self.eval_games)
        num_batches = (self.eval_games + games_per_batch - 1) // games_per_batch

        for batch_idx in range(num_batches):
            batch_size = min(games_per_batch, self.eval_games - batch_idx * games_per_batch)
            if batch_size <= 0:
                break

            batch_results = self._run_eval_batch(
                batch_size,
                opponent_evaluator,
                eval_type,
            )

            wins += batch_results['wins']
            losses += batch_results['losses']
            draws += batch_results['draws']
            total_game_length += batch_results['total_length']
            total_points += batch_results['total_points']

            if not self.running:
                break

        games_played = wins + losses + draws
        if games_played == 0:
            return None

        return EvalResult(
            eval_type=eval_type,
            model_version=model_version,
            games_played=games_played,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / games_played,
            avg_game_length=total_game_length / games_played,
            avg_points_won=total_points / games_played,
            duration_seconds=time.time() - start_time,
            timestamp=time.time(),
        )

    def _run_eval_batch(
        self,
        batch_size: int,
        opponent_evaluator,
        eval_type: str,
    ) -> Dict[str, Any]:
        """Run a batch of evaluation games.

        Args:
            batch_size: Number of parallel games.
            opponent_evaluator: The opponent's evaluator.
            eval_type: Type of evaluation.

        Returns:
            Dict with wins, losses, draws, total_length, total_points.
        """
        key, self._rng_key = jax.random.split(self._rng_key)

        # Initialize environments
        init_keys = jax.random.split(key, batch_size)
        env_states, metadatas = jax.vmap(self._env_init_fn)(init_keys)

        # Initialize evaluator states
        template_state, _ = self._env_init_fn(jax.random.PRNGKey(0))

        # Our model plays as player 0, opponent as player 1
        our_eval_states = self._mcts_evaluator.init_batched(
            batch_size, template_embedding=template_state
        )

        # For opponent, handle both MCTS-style and simple evaluators
        if hasattr(opponent_evaluator, 'init_batched'):
            opp_eval_states = opponent_evaluator.init_batched(
                batch_size, template_embedding=template_state
            )
        else:
            opp_eval_states = jax.vmap(opponent_evaluator.init)(
                jax.random.split(key, batch_size)
            )

        # Track game state
        game_lengths = jnp.zeros(batch_size, dtype=jnp.int32)
        final_rewards = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        done = jnp.zeros(batch_size, dtype=bool)

        max_steps = 500

        for step_idx in range(max_steps):
            if jnp.all(done):
                break

            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, batch_size)

            # Determine which player's turn it is
            current_players = metadatas.cur_player_id

            # Get actions from both evaluators
            our_outputs = self._eval_batch(
                self._mcts_evaluator,
                our_eval_states,
                env_states,
                metadatas,
                self._nn_params,
                step_keys,
            )

            opp_outputs = self._eval_batch_opponent(
                opponent_evaluator,
                opp_eval_states,
                env_states,
                metadatas,
                step_keys,
            )

            # Select action based on current player
            actions = jnp.where(
                current_players[:, None] == 0,
                our_outputs.action[:, None],
                opp_outputs.action[:, None]
            ).squeeze(-1)

            # Step environments
            def step_single(args):
                env_state, metadata, action, step_key, is_done = args
                new_env_state, new_metadata = jax.lax.cond(
                    is_done,
                    lambda _: (env_state, metadata),
                    lambda _: self._env_step_fn(env_state, action, step_key),
                    None
                )
                return new_env_state, new_metadata

            new_env_states, new_metadatas = jax.vmap(step_single)(
                (env_states, metadatas, actions, step_keys, done)
            )

            # Update done mask and rewards
            newly_done = new_metadatas.terminated & ~done
            game_lengths = game_lengths + jnp.where(~done, 1, 0)

            # Capture final rewards when game ends
            final_rewards = jnp.where(
                newly_done[:, None],
                new_metadatas.rewards,
                final_rewards
            )

            done = done | new_metadatas.terminated

            # Update states
            env_states = new_env_states
            metadatas = new_metadatas
            our_eval_states = our_outputs.eval_state
            opp_eval_states = opp_outputs.eval_state

        # Count results (player 0 is our model)
        our_rewards = final_rewards[:, 0]
        wins = int(jnp.sum(our_rewards > 0))
        losses = int(jnp.sum(our_rewards < 0))
        draws = int(jnp.sum(our_rewards == 0))

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'total_length': int(jnp.sum(game_lengths)),
            'total_points': float(jnp.sum(our_rewards)),
        }

    def _eval_batch(
        self,
        evaluator,
        eval_states,
        env_states,
        metadatas,
        params,
        keys,
    ):
        """Evaluate a batch with our MCTS evaluator."""

        def eval_single(eval_state, env_state, metadata, key):
            return evaluator.evaluate(
                key=key,
                eval_state=eval_state,
                env_state=env_state,
                root_metadata=metadata,
                params=params,
                env_step_fn=self._env_step_fn,
            )

        return jax.vmap(eval_single)(eval_states, env_states, metadatas, keys)

    def _eval_batch_opponent(
        self,
        evaluator,
        eval_states,
        env_states,
        metadatas,
        keys,
    ):
        """Evaluate a batch with opponent evaluator."""

        def eval_single(eval_state, env_state, key):
            return evaluator.evaluate(
                key=key,
                eval_state=eval_state,
                env_state=env_state,
            )

        return jax.vmap(eval_single)(eval_states, env_states, keys)

    def _run_loop(self, num_iterations: int = -1) -> Dict[str, Any]:
        """Main evaluation loop.

        Waits for new model versions and runs evaluations.

        Args:
            num_iterations: Maximum eval runs (-1 for infinite).

        Returns:
            Dict with results/statistics from the run.
        """
        # Setup
        print(f"Worker {self.worker_id}: Setting up evaluation environment...")
        self._setup_environment()
        self._setup_neural_network()
        self._setup_mcts_evaluator()
        self._setup_gnubg_evaluator()
        self._setup_random_evaluator()
        self._initialize_params()

        # Start Prometheus metrics server
        metrics_port = self.config.get('metrics_port', 9300)
        start_metrics_server(metrics_port)
        metrics = get_metrics()

        # Register metrics endpoint
        register_metrics_endpoint(
            self.buffer.redis,
            worker_id=self.worker_id,
            worker_type='evaluation',
            port=metrics_port,
            ttl_seconds=60,
        )

        # Set worker info
        metrics.worker_info.labels(worker_id=self.worker_id).info({
            'type': 'evaluation',
            'eval_games': str(self.eval_games),
            'eval_types': ','.join(self.enabled_eval_types),
        })
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='evaluation'
        ).set(1)

        print(f"Worker {self.worker_id}: Starting evaluation loop...")
        print(f"Worker {self.worker_id}: Enabled eval types: {self.enabled_eval_types}")

        start_time = time.time()
        eval_count = 0
        last_check_version = 0

        while self.running:
            if num_iterations >= 0 and eval_count >= num_iterations:
                break

            # Check if there's a new model version
            current_version = self.current_model_version

            if current_version > last_check_version:
                # Queue eval tasks for the new version
                self._queue_eval_tasks(current_version)
                last_check_version = current_version

            # Try to claim an eval task
            task = self._claim_eval_task()

            if task is None:
                # No tasks available, wait and check for model updates
                time.sleep(5.0)

                # Refresh metrics registration
                register_metrics_endpoint(
                    self.buffer.redis,
                    worker_id=self.worker_id,
                    worker_type='evaluation',
                    port=metrics_port,
                    ttl_seconds=60,
                )
                continue

            model_version, eval_type = task
            print(
                f"Worker {self.worker_id}: Running {eval_type} evaluation "
                f"for model v{model_version}"
            )

            # Run the evaluation
            result = self._run_evaluation(model_version, eval_type)

            if result is not None:
                # Store result
                self._complete_eval_task(result)
                eval_count += 1

                # Update metrics
                self._update_metrics(result, metrics)
            else:
                # Failed, remove from in-progress
                task_id = f"v{model_version}:{eval_type}"
                self.buffer.redis.srem(self.EVAL_IN_PROGRESS, task_id)

        # Mark worker as stopped
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='evaluation'
        ).set(0)

        # Cleanup
        self.buffer.close()

        return {
            'status': 'completed',
            'total_evaluations': eval_count,
            'duration_seconds': time.time() - start_time,
        }

    def _update_metrics(self, result: EvalResult, metrics) -> None:
        """Update Prometheus metrics with evaluation result."""
        # Evaluation-specific metrics
        metrics.eval_games_total.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.games_played)

        metrics.eval_wins.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.wins)

        metrics.eval_losses.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.losses)

        metrics.eval_win_rate.labels(
            eval_type=result.eval_type, model_version=str(result.model_version)
        ).set(result.win_rate)

        metrics.eval_duration.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).observe(result.duration_seconds)

        metrics.eval_avg_game_length.labels(
            eval_type=result.eval_type, model_version=str(result.model_version)
        ).set(result.avg_game_length)

        metrics.eval_runs_total.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc()

        # General worker metrics
        metrics.games_total.labels(
            worker_id=self.worker_id, worker_type='evaluation'
        ).inc(result.games_played)

        metrics.model_version.labels(
            worker_id=self.worker_id, worker_type='evaluation'
        ).set(result.model_version)

    def get_eval_stats(self) -> Dict[str, Any]:
        """Get current evaluation statistics.

        Returns:
            Dict with evaluation stats.
        """
        base_stats = self.get_stats()
        base_stats.update({
            'eval_games': self.eval_games,
            'batch_size': self.batch_size,
            'num_simulations': self.num_simulations,
            'enabled_eval_types': self.enabled_eval_types,
            'last_evaluated_version': self._last_evaluated_version,
        })
        return base_stats
