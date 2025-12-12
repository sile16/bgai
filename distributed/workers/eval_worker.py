"""Evaluation worker for distributed training.

This worker evaluates the current model against various baselines
after each new epoch (model version update). Multiple eval workers
can run in parallel, picking up different evaluation types.

Eval types:
- gnubg: Play against GNU Backgammon
- random: Play against random policy
- self_play: Play current model against itself (baseline)
- checkpoint_N: Play against a specific checkpoint version

No Ray dependency - uses Redis for all coordination.
"""

import time
from enum import Enum
from functools import partial
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import chex

from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.evaluator import EvalOutput
from core.common import two_player_game
from core.types import StepMetadata

from .base_worker import BaseWorker, WorkerStats
from ..serialization import deserialize_weights
from ..buffer.redis_buffer import RedisReplayBuffer
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint
from ..coordinator.redis_state import create_state_manager


class EvalType(Enum):
    """Types of evaluation that can be performed."""
    GNUBG = "gnubg"           # Against GNU Backgammon
    RANDOM = "random"         # Against random policy
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


class EvalWorker(BaseWorker):
    """Evaluation worker that evaluates models against baselines.

    Eval workers wait for new model versions and then run evaluation
    games against various opponents. Each worker picks up individual
    eval types from a queue, allowing parallel evaluation.

    No Ray dependency - uses Redis for all coordination.

    Example:
        >>> worker = EvalWorker(config={'eval_games': 100})
        >>> worker.run()

    TODO: Future optimization - distributed eval aggregation:
        Currently one worker runs all games for a (version, eval_type) pair.
        Better approach: set a target total (e.g., 100 games), workers check out
        batches dynamically based on their local hardware batch size.
        - Worker 1 (batch=16): checks out 16, finishes, checks out 16 more...
        - Worker 2 (batch=50): checks out 50, finishes fast, grabs remaining 34
        - Use Redis INCRBY for atomic game counter
        - Aggregate partial results when target reached
        This maximizes hardware utilization across heterogeneous nodes.
    """

    # Redis keys for eval coordination
    EVAL_QUEUE = "bgai:eval:queue"
    EVAL_RESULTS = "bgai:eval:results"
    EVAL_IN_PROGRESS = "bgai:eval:in_progress"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
    ):
        """Initialize the eval worker.

        Args:
            config: Configuration dict with keys:
                - eval_games: Games per evaluation (default: 100)
                - batch_size: Number of parallel games (default: 16)
                - num_simulations: MCTS simulations per move (default: 200)
                - max_nodes: Maximum MCTS tree nodes (default: 800)
                - eval_types: List of eval types to run (default: all)
                - redis_host/port/password: Redis connection info
            worker_id: Optional unique worker ID. Auto-generated if not provided.
        """
        super().__init__(config, worker_id)

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

        # MLflow configuration
        self.mlflow_tracking_uri = self.config.get('mlflow_tracking_uri')
        self.mlflow_experiment_name = self.config.get('mlflow_experiment_name', 'bgai-training')
        self._mlflow_run = None

        # Create state manager for getting run ID
        self.state = create_state_manager({
            'redis_host': redis_host,
            'redis_port': redis_port,
            'redis_password': redis_password,
        })

    @property
    def worker_type(self) -> str:
        return 'eval'

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

            Value head outputs 4 logits (use sigmoid, not softmax):
            [win, gammon_win|win, gammon_loss|loss, backgammon_rate]
            """
            num_actions: int
            num_hidden: int = 256
            num_blocks: int = 6
            value_head_out_size: int = 4  # 4-way conditional value head

            @nn.compact
            def __call__(self, x, train: bool = False):  # noqa: ARG002 - train required by interface
                del train  # unused but required by turbozero interface
                x = nn.Dense(self.num_hidden)(x)
                x = nn.LayerNorm()(x)
                x = nn.relu(x)

                for _ in range(self.num_blocks):
                    x = ResidualDenseBlock(self.num_hidden)(x)

                policy_logits = nn.Dense(self.num_actions)(x)
                # 4-way value head: outputs logits, converted to probs by evaluator
                value_logits = nn.Dense(self.value_head_out_size)(x)
                return policy_logits, value_logits

        # Use network config from YAML, with fallback to defaults
        num_hidden = self.config.get('network_hidden_dim', 256)
        num_blocks = self.config.get('network_num_blocks', 6)
        self._nn_model = ResNetTurboZero(
            self._env.num_actions,
            num_hidden=num_hidden,
            num_blocks=num_blocks
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
        """Set up the GNUBG evaluator with configurable settings."""
        try:
            from bgai.gnubg_evaluator import GnubgEvaluator, GnubgSettings

            # Get gnubg settings from config
            gnubg_config = self.config.get('gnubg', {})
            self._gnubg_evaluator = GnubgEvaluator(self._env, settings=gnubg_config)

            # Log settings
            settings = self._gnubg_evaluator.get_settings()
            print(f"Worker {self.worker_id}: GNUBG evaluator initialized")
            print(f"  ply={settings.ply}, shortcuts={settings.shortcuts}, "
                  f"osdb={settings.osdb}, move_filters={settings.move_filters}")
        except ImportError as e:
            print(f"Worker {self.worker_id}: GNUBG not available: {e}")
            self._gnubg_evaluator = None

    def _setup_random_evaluator(self) -> None:
        """Set up a random action evaluator."""
        # Simple random evaluator that samples uniformly from legal actions
        # Compatible with TurboZero's two_player_game interface

        class RandomEvaluator:
            def __init__(self, env):
                self.env = env
                self._stochastic_probs = jnp.asarray(
                    env.stochastic_action_probs, dtype=jnp.float32
                )
                self._num_stochastic_actions = int(self._stochastic_probs.shape[0])
                self._num_actions = env.num_actions
                # Required by two_player_game
                self.discount = 1.0

            def init(self, *args, **kwargs):
                return jnp.array(0, dtype=jnp.int32)

            def reset(self, state):
                return self.init()

            def get_value(self, state):
                return jnp.array(0.0, dtype=jnp.float32)

            def step(self, state, action):
                return state

            def evaluate(self, key, eval_state, env_state, root_metadata=None,
                        params=None, env_step_fn=None, **kwargs):
                # Accept but ignore extra args for two_player_game compatibility
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

    def _setup_mlflow(self, max_retries: int = 5, retry_delay: float = 3.0) -> None:
        """Set up MLflow tracking if configured.

        Retries attachment to handle race condition where eval worker starts
        before training worker has created/updated the MLflow run.

        Args:
            max_retries: Maximum number of retry attempts.
            retry_delay: Seconds to wait between retries.
        """
        if not self.mlflow_tracking_uri:
            print(f"Worker {self.worker_id}: MLflow not configured, skipping")
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)

            # Retry loop to handle race condition with training worker
            for attempt in range(max_retries):
                # Get run ID from Redis (shared with training worker)
                run_id = self.state.get_run_id()

                if run_id:
                    # Try to attach to existing run
                    try:
                        self._mlflow_run = mlflow.start_run(run_id=run_id, log_system_metrics=True)
                        print(f"Worker {self.worker_id}: Attached to MLflow run {run_id} (system metrics enabled)")
                        return  # Success
                    except Exception as resume_error:
                        if 'RESOURCE_DOES_NOT_EXIST' in str(resume_error):
                            # Run ID in Redis doesn't exist in MLflow yet
                            # Training worker may be creating a new run
                            if attempt < max_retries - 1:
                                print(
                                    f"Worker {self.worker_id}: Run {run_id} not found, "
                                    f"waiting for training worker... (attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(retry_delay)
                                continue
                        print(f"Worker {self.worker_id}: Could not attach to run {run_id}: {resume_error}")
                        self._mlflow_run = None
                        return
                else:
                    # No run ID yet, training worker may not have started
                    if attempt < max_retries - 1:
                        print(
                            f"Worker {self.worker_id}: No MLflow run ID in Redis, "
                            f"waiting for training worker... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(retry_delay)
                        continue
                    print(f"Worker {self.worker_id}: No MLflow run ID found in Redis after {max_retries} attempts")
                    self._mlflow_run = None
                    return

        except Exception as e:
            print(f"Worker {self.worker_id}: MLflow setup error: {e}")
            self._mlflow_run = None

    def _log_mlflow_eval_metrics(self, result: 'EvalResult') -> None:
        """Log evaluation metrics to MLflow.

        Args:
            result: EvalResult from an evaluation run.
        """
        if self._mlflow_run is None:
            return

        try:
            import mlflow

            # Use eval_type as prefix for metrics
            prefix = f"eval_{result.eval_type}"

            metrics = {
                f'{prefix}_win_rate': result.win_rate,
                f'{prefix}_games_played': result.games_played,
                f'{prefix}_wins': result.wins,
                f'{prefix}_losses': result.losses,
                f'{prefix}_draws': result.draws,
                f'{prefix}_avg_game_length': result.avg_game_length,
                f'{prefix}_avg_points_won': result.avg_points_won,
                f'{prefix}_duration_seconds': result.duration_seconds,
            }

            # Log with model version as step for tracking over training
            mlflow.log_metrics(metrics, step=result.model_version)

            print(
                f"Worker {self.worker_id}: Logged MLflow metrics for "
                f"{result.eval_type} v{result.model_version}"
            )

        except Exception as e:
            print(f"Worker {self.worker_id}: MLflow logging error: {e}")

    def _log_mlflow_gnubg_params(self) -> None:
        """Log GNUBG settings as MLflow params (once per run).

        This records the GNUBG configuration used for evaluation so we can
        compare results across runs with different settings.
        """
        if self._mlflow_run is None or self._gnubg_evaluator is None:
            return

        try:
            import mlflow

            # Get settings dict from evaluator
            settings_dict = self._gnubg_evaluator.get_settings_dict()

            # Convert move_filters list to string for MLflow params
            settings_dict['gnubg_move_filters'] = str(settings_dict['gnubg_move_filters'])

            mlflow.log_params(settings_dict)
            print(f"Worker {self.worker_id}: Logged GNUBG settings to MLflow")

        except Exception as e:
            # Params may already be logged from a previous worker
            if "already logged" not in str(e).lower():
                print(f"Worker {self.worker_id}: MLflow GNUBG params logging: {e}")

    def _initialize_params(self) -> None:
        """Initialize neural network parameters from Redis."""
        result = self.get_current_model_weights()

        if result is not None:
            weights_bytes, version = result
            params_dict = deserialize_weights(weights_bytes)
            self._nn_params = params_dict
            self.current_model_version = version
            print(f"Worker {self.worker_id}: Loaded model version {version}")
        else:
            # Initialize random weights
            key = jax.random.PRNGKey(42)
            sample_state, _ = self._env_init_fn(key)
            sample_obs = self._state_to_nn_input_fn(sample_state)
            variables = self._nn_model.init(key, sample_obs[None, ...])
            self._nn_params = {'params': variables['params']}
            print(f"Worker {self.worker_id}: Initialized random weights")

    def _check_and_update_model(self) -> bool:
        """Check for model updates and load if available.

        Returns:
            True if model was updated.
        """
        result = self.get_model_weights()
        if result is not None:
            weights_bytes, version = result
            self._nn_params = deserialize_weights(weights_bytes)
            print(f"Worker {self.worker_id}: Updated to model version {version}")
            return True
        return False

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
            result = self.get_current_model_weights()
            if result is not None:
                weights_bytes, version = result
                self._nn_params = deserialize_weights(weights_bytes)
                self.current_model_version = version

        # Route to appropriate evaluation method based on eval_type
        if eval_type == EvalType.GNUBG.value:
            if self._gnubg_evaluator is None:
                print(f"Worker {self.worker_id}: GNUBG not available, skipping")
                return None
            # GNUBG uses pure_callback, not JIT-compatible with two_player_game
            results = self._run_gnubg_eval()
        elif eval_type == EvalType.RANDOM.value:
            # Use TurboZero's two_player_game (JIT-compatible)
            results = self._run_two_player_eval(self._random_evaluator, None)
        else:
            print(f"Worker {self.worker_id}: Unknown eval type: {eval_type}")
            return None

        if results is None:
            return None

        wins = results['wins']
        losses = results['losses']
        draws = results['draws']
        total_game_length = results['total_length']
        total_points = results['total_points']

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

    def _run_two_player_eval(
        self,
        opponent_evaluator,
        opponent_params: Optional[chex.ArrayTree],
    ) -> Dict[str, Any]:
        """Run evaluation games using TurboZero's two_player_game.

        Uses jax.lax.scan for efficient JIT-compiled game loops with proper
        evaluator state management (calls evaluator.step() after each action).

        Args:
            opponent_evaluator: The opponent's evaluator.
            opponent_params: Parameters for opponent (None for random).

        Returns:
            Dict with wins, losses, draws, total_length, total_points.
        """
        wins, losses, draws = 0, 0, 0
        total_game_length = 0
        total_points = 0.0

        max_steps = 500

        # Run games one at a time (two_player_game handles single games)
        for game_idx in range(self.eval_games):
            if not self.running:
                break

            key, self._rng_key = jax.random.split(self._rng_key)

            # Run a single game using TurboZero's two_player_game
            # Our MCTS evaluator is player 1, opponent is player 2
            outcomes, frames, p_ids = two_player_game(
                key=key,
                evaluator_1=self._mcts_evaluator,
                evaluator_2=opponent_evaluator,
                params_1=self._nn_params,
                params_2=opponent_params if opponent_params is not None else self._nn_params,
                env_step_fn=self._env_step_fn,
                env_init_fn=self._env_init_fn,
                max_steps=max_steps,
            )

            # outcomes[0] is evaluator_1's outcome, outcomes[1] is evaluator_2's
            # p_ids tells us which player ID each evaluator got assigned
            our_outcome = float(outcomes[0])
            game_length = int(jnp.sum(~frames.completed))

            if our_outcome > 0:
                wins += 1
            elif our_outcome < 0:
                losses += 1
            else:
                draws += 1

            total_game_length += game_length
            total_points += our_outcome

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'total_length': total_game_length,
            'total_points': total_points,
        }

    def _run_gnubg_eval(self) -> Dict[str, Any]:
        """Run evaluation games against GNUBG.

        Uses a Python loop since GNUBG evaluator uses jax.pure_callback
        which is not compatible with jax.lax.scan in two_player_game.

        This method properly steps both evaluators after each action.

        Returns:
            Dict with wins, losses, draws, total_length, total_points.
        """
        wins, losses, draws = 0, 0, 0
        total_game_length = 0
        total_points = 0.0

        max_steps = 500

        for game_idx in range(self.eval_games):
            if not self.running:
                break

            key, self._rng_key = jax.random.split(self._rng_key)

            # Initialize game
            key, init_key = jax.random.split(key)
            env_state, metadata = self._env_init_fn(init_key)

            # Initialize evaluator states
            our_eval_state = self._mcts_evaluator.init(template_embedding=env_state)
            opp_eval_state = self._gnubg_evaluator.init(template_embedding=env_state)

            # Randomly assign which evaluator plays which color
            key, turn_key = jax.random.split(key)
            first_player = int(jax.random.randint(turn_key, (), 0, 2))
            we_are_player_0 = (first_player == 0)

            game_length = 0
            done = False

            while not done and game_length < max_steps:
                key, step_key = jax.random.split(key)
                current_player = int(metadata.cur_player_id)

                # Determine which evaluator should move
                if (current_player == 0) == we_are_player_0:
                    # Our turn (MCTS)
                    output = self._mcts_evaluator.evaluate(
                        key=step_key,
                        eval_state=our_eval_state,
                        env_state=env_state,
                        root_metadata=metadata,
                        params=self._nn_params,
                        env_step_fn=self._env_step_fn,
                    )
                    action = output.action
                    # Step our evaluator and the opponent's evaluator
                    our_eval_state = self._mcts_evaluator.step(output.eval_state, action)
                    opp_eval_state = self._gnubg_evaluator.step(opp_eval_state, action)
                else:
                    # Opponent's turn (GNUBG)
                    output = self._gnubg_evaluator.evaluate(
                        key=step_key,
                        eval_state=opp_eval_state,
                        env_state=env_state,
                    )
                    action = output.action
                    # Step both evaluators
                    opp_eval_state = self._gnubg_evaluator.step(output.eval_state, action)
                    our_eval_state = self._mcts_evaluator.step(our_eval_state, action)

                # Step environment
                env_state, metadata = self._env_step_fn(env_state, action, step_key)
                game_length += 1

                if metadata.terminated:
                    done = True

            # Determine outcome from our perspective
            if we_are_player_0:
                our_reward = float(metadata.rewards[0])
            else:
                our_reward = float(metadata.rewards[1])

            if our_reward > 0:
                wins += 1
            elif our_reward < 0:
                losses += 1
            else:
                draws += 1

            total_game_length += game_length
            total_points += our_reward

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'total_length': total_game_length,
            'total_points': total_points,
        }

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

        # Set up MLflow tracking
        self._setup_mlflow()

        # Log GNUBG settings to MLflow (if gnubg eval is enabled)
        if EvalType.GNUBG.value in self.enabled_eval_types:
            self._log_mlflow_gnubg_params()

        # Start Prometheus metrics server
        metrics_port_config = self.config.get('metrics_port', 9300)
        metrics_port = start_metrics_server(metrics_port_config)
        if metrics_port is None:
            print(f"Worker {self.worker_id}: Failed to start metrics server")
            metrics_port = metrics_port_config  # Fallback for registration
        metrics = get_metrics()

        # Pre-initialize evaluation metrics so Grafana shows series
        # even before the first evaluation completes.
        current_version_str = str(self.current_model_version)
        for eval_type in self.enabled_eval_types:
            metrics.eval_win_rate.labels(
                eval_type=eval_type, model_version=current_version_str
            ).set(0.0)
            metrics.eval_avg_game_length.labels(
                eval_type=eval_type, model_version=current_version_str
            ).set(0.0)
            metrics.eval_games_total.labels(
                worker_id=self.worker_id, eval_type=eval_type
            ).inc(0)
            metrics.eval_wins.labels(
                worker_id=self.worker_id, eval_type=eval_type
            ).inc(0)
            metrics.eval_losses.labels(
                worker_id=self.worker_id, eval_type=eval_type
            ).inc(0)
            metrics.eval_runs_total.labels(
                worker_id=self.worker_id, eval_type=eval_type
            ).inc(0)

        # Register metrics endpoint (use actual bound port)
        register_metrics_endpoint(
            self.buffer.redis,
            worker_id=self.worker_id,
            worker_type='eval',
            port=metrics_port,
            ttl_seconds=60,
        )

        # Set worker info
        metrics.worker_info.labels(worker_id=self.worker_id).info({
            'type': 'eval',
            'eval_games': str(self.eval_games),
            'eval_types': ','.join(self.enabled_eval_types),
        })
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(1)

        # Set configuration metrics (for correlating with memory usage)
        metrics.worker_batch_size.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(self.batch_size)
        metrics.worker_num_simulations.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(self.num_simulations)
        metrics.worker_max_nodes.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(self.max_nodes)

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
                self._check_and_update_model()
                time.sleep(5.0)

                # Refresh metrics registration
                register_metrics_endpoint(
                    self.buffer.redis,
                    worker_id=self.worker_id,
                    worker_type='eval',
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

                # Log to MLflow
                self._log_mlflow_eval_metrics(result)
            else:
                # Failed, remove from in-progress
                task_id = f"v{model_version}:{eval_type}"
                self.buffer.redis.srem(self.EVAL_IN_PROGRESS, task_id)

        # Mark worker as stopped
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(0)

        # End MLflow run (don't call end_run as we're sharing with training worker)
        # Just clear our reference
        self._mlflow_run = None

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
            worker_id=self.worker_id, worker_type='eval'
        ).inc(result.games_played)

        metrics.model_version.labels(
            worker_id=self.worker_id, worker_type='eval'
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
