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

import gc
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
from ..metrics import get_metrics, start_metrics_server, register_metrics_endpoint, WorkerPhase, WorkerState
from ..coordinator.redis_state import create_state_manager


class EvalType(Enum):
    """Types of evaluation that can be performed."""
    GNUBG = "gnubg"           # Against GNU Backgammon
    RANDOM = "random"         # Against random policy
    GREEDY = "greedy"         # Against greedy policy (no MCTS)
    SELF_PLAY = "self_play"   # Current model vs itself (baseline)


@dataclass
class EvalResult:
    """Result of an evaluation run.

    Backgammon scoring:
    - Normal win/loss: 1 point
    - Gammon (opponent has no pieces off): 2 points
    - Backgammon (gammon + opponent has piece in bar/home): 3 points

    Points are tracked from our perspective: positive = we won, negative = we lost.
    """
    eval_type: str
    model_version: int
    games_played: int
    wins: int              # Total wins (1, 2, or 3 points)
    losses: int            # Total losses (-1, -2, or -3 points)
    draws: int
    # Detailed breakdown by point value
    wins_single: int       # 1-point wins (normal)
    wins_gammon: int       # 2-point wins (gammon)
    wins_backgammon: int   # 3-point wins (backgammon)
    losses_single: int     # 1-point losses
    losses_gammon: int     # 2-point losses
    losses_backgammon: int # 3-point losses
    win_rate: float
    avg_game_length: float
    avg_points_won: float  # Average points won per game (can be negative)
    total_points: int      # Total points scored (positive - negative)
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
    EVAL_PROGRESS = "bgai:eval:progress"  # Tracks games completed per (version, eval_type)

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

        # Prometheus metrics discovery registration. Eval runs can be long, so
        # refresh the Redis heartbeat periodically during evaluations to avoid
        # disappearing from Prometheus/Grafana.
        self._metrics_port: Optional[int] = None
        self._metrics_ttl_seconds = 60
        self._metrics_refresh_interval_seconds = 15.0
        self._last_metrics_refresh_time = 0.0

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
        self._self_play_opponent_evaluator = None  # Reused for self-play evals

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

    def _refresh_metrics_registration(self, force: bool = False) -> None:
        if self._metrics_port is None:
            return

        now = time.time()
        if (not force) and (now - self._last_metrics_refresh_time) < self._metrics_refresh_interval_seconds:
            return

        try:
            register_metrics_endpoint(
                self.buffer.redis,
                worker_id=self.worker_id,
                worker_type='eval',
                port=self._metrics_port,
                ttl_seconds=self._metrics_ttl_seconds,
            )
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to refresh metrics registration: {e}")
        else:
            self._last_metrics_refresh_time = now

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
            persist_tree=False,  # Don't persist tree between games to prevent memory growth
        )

    def _get_self_play_opponent_evaluator(self) -> StochasticMCTS:
        """Get (or create) the self-play opponent evaluator.

        Reuses a single evaluator instance to avoid memory leaks from
        creating new JIT-compiled functions on each evaluation run.
        """
        if self._self_play_opponent_evaluator is None:
            if self._nn_model is None:
                self._setup_neural_network()

            eval_fn = make_nn_eval_fn(self._nn_model, self._state_to_nn_input_fn)
            self._self_play_opponent_evaluator = StochasticMCTS(
                eval_fn=eval_fn,
                stochastic_action_probs=self._env.stochastic_action_probs,
                num_iterations=self.num_simulations,
                max_nodes=self.max_nodes,
                branching_factor=self._env.num_actions,
                action_selector=PUCTSelector(),
                temperature=0.1,
                persist_tree=False,  # Don't persist tree between games to prevent memory growth
            )
        return self._self_play_opponent_evaluator

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
                        self._mlflow_run = mlflow.start_run(run_id=run_id, log_system_metrics=False)
                        print(f"Worker {self.worker_id}: Attached to MLflow run {run_id}")
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
                # Detailed point breakdown
                f'{prefix}_wins_single': result.wins_single,
                f'{prefix}_wins_gammon': result.wins_gammon,
                f'{prefix}_wins_backgammon': result.wins_backgammon,
                f'{prefix}_losses_single': result.losses_single,
                f'{prefix}_losses_gammon': result.losses_gammon,
                f'{prefix}_losses_backgammon': result.losses_backgammon,
                f'{prefix}_total_points': result.total_points,
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

    def _get_eval_progress_key(self, model_version: int, eval_type: str) -> str:
        """Get Redis key for tracking eval progress."""
        return f"{self.EVAL_PROGRESS}:v{model_version}:{eval_type}"

    def _claim_eval_batch(self, model_version: int, eval_type: str) -> Optional[int]:
        """Atomically claim a batch of games to evaluate.

        Uses Redis INCRBY to atomically increment the games counter.
        Returns the number of games to run in this batch, or None if target reached.

        Args:
            model_version: Model version to evaluate.
            eval_type: Type of evaluation (gnubg, random, etc).

        Returns:
            Number of games to run in this batch, or None if target already reached.
        """
        progress_key = self._get_eval_progress_key(model_version, eval_type)

        # Atomically check and increment
        # Using WATCH/MULTI/EXEC for atomic check-and-increment
        pipe = self.buffer.redis.pipeline()
        try:
            pipe.watch(progress_key)
            current = pipe.get(progress_key)
            current_games = int(current) if current else 0

            if current_games >= self.eval_games:
                pipe.unwatch()
                return None  # Target already reached

            # Calculate batch size (don't exceed target)
            remaining = self.eval_games - current_games
            batch = min(self.batch_size, remaining)

            # Atomically increment
            pipe.multi()
            pipe.incrby(progress_key, batch)
            pipe.expire(progress_key, 3600)  # 1 hour TTL
            pipe.execute()

            return batch
        except Exception as e:
            print(f"Worker {self.worker_id}: Redis claim error: {e}", flush=True)
            try:
                pipe.unwatch()
            except Exception:
                pass
            return None

    def _get_eval_progress(self, model_version: int, eval_type: str) -> int:
        """Get current global progress for an eval task.

        Returns:
            Number of games already completed/claimed globally.
        """
        progress_key = self._get_eval_progress_key(model_version, eval_type)
        current = self.buffer.redis.get(progress_key)
        return int(current) if current else 0

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

    def _emit_batch_result(self, result: EvalResult, is_final: bool = False) -> None:
        """Emit batch results immediately to Prometheus and optionally store in Redis.

        Called after each batch completes. Prometheus counters aggregate across workers.

        Args:
            result: EvalResult from the batch.
            is_final: Whether this completes the eval for this (version, eval_type).
        """
        task_id = f"v{result.model_version}:{result.eval_type}"

        # Check global progress
        global_progress = self._get_eval_progress(result.model_version, result.eval_type)

        # Print batch completion
        print(
            f"Worker {self.worker_id}: Batch complete {result.eval_type} v{result.model_version}: "
            f"{result.games_played} games, win_rate={result.win_rate:.2%}, points={result.total_points} "
            f"(W:{result.wins_single}/{result.wins_gammon}/{result.wins_backgammon} "
            f"L:{result.losses_single}/{result.losses_gammon}/{result.losses_backgammon}) "
            f"[global: {global_progress}/{self.eval_games}]",
            flush=True
        )

        # Always update Prometheus metrics immediately (they aggregate across workers)
        # The _update_metrics call in _run_loop handles this

        if is_final:
            # Store aggregated result for this worker's contribution
            result_key = f"{self.EVAL_RESULTS}:{task_id}:{self.worker_id}"
            result_data = {
                'eval_type': result.eval_type,
                'model_version': result.model_version,
                'games_played': result.games_played,
                'wins': result.wins,
                'losses': result.losses,
                'draws': result.draws,
                'wins_single': result.wins_single,
                'wins_gammon': result.wins_gammon,
                'wins_backgammon': result.wins_backgammon,
                'losses_single': result.losses_single,
                'losses_gammon': result.losses_gammon,
                'losses_backgammon': result.losses_backgammon,
                'win_rate': result.win_rate,
                'avg_game_length': result.avg_game_length,
                'avg_points_won': result.avg_points_won,
                'total_points': result.total_points,
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
                {f"{task_id}:{self.worker_id}": result.model_version}
            )

    def _complete_eval_task(self, result: EvalResult) -> None:
        """Mark an evaluation task as complete and store results.

        DEPRECATED: Use _emit_batch_result for new batch-based approach.
        Kept for compatibility with legacy single-worker eval completion.
        """
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
            'wins_single': result.wins_single,
            'wins_gammon': result.wins_gammon,
            'wins_backgammon': result.wins_backgammon,
            'losses_single': result.losses_single,
            'losses_gammon': result.losses_gammon,
            'losses_backgammon': result.losses_backgammon,
            'win_rate': result.win_rate,
            'avg_game_length': result.avg_game_length,
            'avg_points_won': result.avg_points_won,
            'total_points': result.total_points,
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

        # Enhanced print with point breakdown
        print(
            f"Worker {self.worker_id}: Completed {result.eval_type} v{result.model_version}: "
            f"win_rate={result.win_rate:.2%}, points={result.total_points} "
            f"(W:{result.wins_single}/{result.wins_gammon}/{result.wins_backgammon} "
            f"L:{result.losses_single}/{result.losses_gammon}/{result.losses_backgammon})",
            flush=True
        )

    def _run_evaluation(
        self,
        model_version: int,
        eval_type: str,
        num_games: Optional[int] = None,
    ) -> Optional[EvalResult]:
        """Run evaluation games against a specific opponent.

        Args:
            model_version: Model version to evaluate.
            eval_type: Type of evaluation to run.
            num_games: Number of games to run (defaults to self.eval_games for legacy).

        Returns:
            EvalResult or None if evaluation failed.
        """
        start_time = time.time()
        games_to_run = num_games if num_games is not None else self.eval_games

        # Ensure we have the right model version
        if self.current_model_version != model_version:
            result = self.get_current_model_weights()
            if result is not None:
                weights_bytes, version = result
                self._nn_params = deserialize_weights(weights_bytes)
                self.current_model_version = version

        eval_type_normalized = eval_type.replace("-", "_").strip().lower()

        # Route to appropriate evaluation method based on eval_type
        if eval_type == EvalType.GNUBG.value:
            if self._gnubg_evaluator is None:
                print(f"Worker {self.worker_id}: GNUBG not available, skipping")
                return None
            # GNUBG uses pure_callback, not JIT-compatible with two_player_game
            results = self._run_gnubg_eval(num_games=games_to_run)
        elif eval_type_normalized == EvalType.RANDOM.value:
            # Use TurboZero's two_player_game (JIT-compatible)
            results = self._run_two_player_eval(self._random_evaluator, None, num_games=games_to_run)
        elif eval_type_normalized == EvalType.SELF_PLAY.value:
            opponent = self._get_self_play_opponent_evaluator()
            results = self._run_two_player_eval(opponent, self._nn_params, num_games=games_to_run)
        else:
            print(f"Worker {self.worker_id}: Unknown eval type: {eval_type}")
            return None

        if results is None:
            return None

        wins = results['wins']
        losses = results['losses']
        draws = results['draws']
        wins_single = results.get('wins_single', 0)
        wins_gammon = results.get('wins_gammon', 0)
        wins_backgammon = results.get('wins_backgammon', 0)
        losses_single = results.get('losses_single', 0)
        losses_gammon = results.get('losses_gammon', 0)
        losses_backgammon = results.get('losses_backgammon', 0)
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
            wins_single=wins_single,
            wins_gammon=wins_gammon,
            wins_backgammon=wins_backgammon,
            losses_single=losses_single,
            losses_gammon=losses_gammon,
            losses_backgammon=losses_backgammon,
            win_rate=wins / games_played,
            avg_game_length=total_game_length / games_played,
            avg_points_won=total_points / games_played,
            total_points=int(total_points),
            duration_seconds=time.time() - start_time,
            timestamp=time.time(),
        )

    def _get_jitted_game_fn(self, opponent_evaluator):
        """Get a JIT-compiled game function for the given opponent.

        Caches compilation per opponent type to prevent memory leaks from
        repeated JIT tracing.
        """
        opponent_type = type(opponent_evaluator).__name__

        if not hasattr(self, '_jitted_game_fns'):
            self._jitted_game_fns = {}

        if opponent_type not in self._jitted_game_fns:
            # Create a JIT-compiled version of two_player_game for this opponent type
            @jax.jit
            def play_game(key, params_1, params_2):
                return two_player_game(
                    key=key,
                    evaluator_1=self._mcts_evaluator,
                    evaluator_2=opponent_evaluator,
                    params_1=params_1,
                    params_2=params_2,
                    env_step_fn=self._env_step_fn,
                    env_init_fn=self._env_init_fn,
                    max_steps=500,
                )
            self._jitted_game_fns[opponent_type] = play_game

        return self._jitted_game_fns[opponent_type]

    def _run_two_player_eval(
        self,
        opponent_evaluator,
        opponent_params: Optional[chex.ArrayTree],
        num_games: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run evaluation games using TurboZero's two_player_game.

        Uses jax.lax.scan for efficient JIT-compiled game loops with proper
        evaluator state management (calls evaluator.step() after each action).

        Args:
            opponent_evaluator: The opponent's evaluator.
            opponent_params: Parameters for opponent (None for random).
            num_games: Number of games to run (defaults to self.eval_games).

        Returns:
            Dict with detailed win/loss breakdown by point value.
        """
        wins, losses, draws = 0, 0, 0
        wins_single, wins_gammon, wins_backgammon = 0, 0, 0
        losses_single, losses_gammon, losses_backgammon = 0, 0, 0
        total_game_length = 0
        total_points = 0.0
        start_time = time.time()
        opponent_name = type(opponent_evaluator).__name__
        games_to_run = num_games if num_games is not None else self.eval_games

        # Get JIT-compiled game function for this opponent type
        play_game = self._get_jitted_game_fn(opponent_evaluator)
        params_2 = opponent_params if opponent_params is not None else self._nn_params

        # Run games one at a time (two_player_game handles single games)
        for game_idx in range(games_to_run):
            if not self.running:
                break
            self._refresh_metrics_registration()

            # Progress logging every 10 games or every game if games_to_run < 20
            log_interval = max(1, games_to_run // 10)
            if game_idx > 0 and game_idx % log_interval == 0:
                elapsed = time.time() - start_time
                games_per_sec = game_idx / elapsed
                eta = (games_to_run - game_idx) / games_per_sec if games_per_sec > 0 else 0
                print(
                    f"Worker {self.worker_id}: {opponent_name} progress {game_idx}/{games_to_run} "
                    f"({wins}W/{losses}L, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                    flush=True
                )

            key, self._rng_key = jax.random.split(self._rng_key)

            # Run a single game using cached JIT-compiled function
            outcomes, frames, p_ids = play_game(key, self._nn_params, params_2)

            # outcomes[0] is evaluator_1's outcome, outcomes[1] is evaluator_2's
            # p_ids tells us which player ID each evaluator got assigned
            our_outcome = float(outcomes[0])
            game_length = int(jnp.sum(~frames.completed))

            # Explicitly delete large objects to free memory
            del frames, p_ids, outcomes, key

            # Categorize by point value (1=single, 2=gammon, 3=backgammon)
            points = int(round(abs(our_outcome)))
            if our_outcome > 0:
                wins += 1
                if points == 1:
                    wins_single += 1
                elif points == 2:
                    wins_gammon += 1
                elif points >= 3:
                    wins_backgammon += 1
            elif our_outcome < 0:
                losses += 1
                if points == 1:
                    losses_single += 1
                elif points == 2:
                    losses_gammon += 1
                elif points >= 3:
                    losses_backgammon += 1
            else:
                draws += 1

            total_game_length += game_length
            total_points += our_outcome

        # Force garbage collection after evaluation batch to prevent memory buildup
        gc.collect()

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'wins_single': wins_single,
            'wins_gammon': wins_gammon,
            'wins_backgammon': wins_backgammon,
            'losses_single': losses_single,
            'losses_gammon': losses_gammon,
            'losses_backgammon': losses_backgammon,
            'total_length': total_game_length,
            'total_points': total_points,
        }

    def _run_gnubg_eval(self, num_games: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation games against GNUBG.

        Uses a Python loop since GNUBG evaluator uses jax.pure_callback
        which is not compatible with jax.lax.scan in two_player_game.

        This method properly steps both evaluators after each action.

        Args:
            num_games: Number of games to run (defaults to self.eval_games).

        Returns:
            Dict with detailed win/loss breakdown by point value.
        """
        wins, losses, draws = 0, 0, 0
        wins_single, wins_gammon, wins_backgammon = 0, 0, 0
        losses_single, losses_gammon, losses_backgammon = 0, 0, 0
        total_game_length = 0
        total_points = 0.0

        max_steps = 500
        start_time = time.time()
        games_to_run = num_games if num_games is not None else self.eval_games

        for game_idx in range(games_to_run):
            if not self.running:
                break
            self._refresh_metrics_registration()

            # Progress logging every game for gnubg (it's slow)
            if game_idx > 0:
                elapsed = time.time() - start_time
                games_per_sec = game_idx / elapsed
                eta = (games_to_run - game_idx) / games_per_sec if games_per_sec > 0 else 0
                print(
                    f"Worker {self.worker_id}: GNUBG progress {game_idx}/{games_to_run} "
                    f"({wins}W/{losses}L, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)",
                    flush=True
                )

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
            last_refresh_time = time.time()

            while not done and game_length < max_steps:
                if time.time() - last_refresh_time >= self._metrics_refresh_interval_seconds:
                    self._refresh_metrics_registration(force=True)
                    last_refresh_time = time.time()

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

            # Categorize by point value (1=single, 2=gammon, 3=backgammon)
            points = int(round(abs(our_reward)))
            if our_reward > 0:
                wins += 1
                if points == 1:
                    wins_single += 1
                elif points == 2:
                    wins_gammon += 1
                elif points >= 3:
                    wins_backgammon += 1
            elif our_reward < 0:
                losses += 1
                if points == 1:
                    losses_single += 1
                elif points == 2:
                    losses_gammon += 1
                elif points >= 3:
                    losses_backgammon += 1
            else:
                draws += 1

            total_game_length += game_length
            total_points += our_reward

            # Explicitly delete large objects to free memory
            del our_eval_state, opp_eval_state, env_state, metadata, output

        # Force garbage collection after evaluation batch to prevent memory buildup
        gc.collect()

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'wins_single': wins_single,
            'wins_gammon': wins_gammon,
            'wins_backgammon': wins_backgammon,
            'losses_single': losses_single,
            'losses_gammon': losses_gammon,
            'losses_backgammon': losses_backgammon,
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
        # Start Prometheus metrics server early so the dashboard can show startup/compile state.
        metrics_port_config = self.config.get('metrics_port', 9300)
        metrics_port = start_metrics_server(metrics_port_config)
        if metrics_port is None:
            print(f"Worker {self.worker_id}: Failed to start metrics server")
            metrics_port = metrics_port_config  # Fallback for registration
        metrics = get_metrics()
        self._metrics_port = metrics_port

        # Register metrics endpoint (use actual bound port)
        self._refresh_metrics_registration(force=True)

        metrics.worker_info.labels(worker_id=self.worker_id).info({
            'type': 'eval',
            'eval_games': str(self.eval_games),
            'eval_types': ','.join(self.enabled_eval_types),
            'batch_size': str(self.batch_size),
            'device_type': 'cuda' if self.device_info.is_cuda else 'metal' if self.device_info.is_metal else 'cpu',
            'device_kind': str(self.device_info.device_kind),
        })
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(1)
        metrics.worker_phase.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(WorkerPhase.IDLE)
        metrics.worker_state.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(WorkerState.COMPILING)
        metrics.worker_steps_per_second.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(0.0)

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
        device_type = 'cuda' if self.device_info.is_cuda else 'metal' if self.device_info.is_metal else 'cpu'
        metrics.worker_device_type.labels(
            worker_id=self.worker_id, worker_type='eval', device_type=device_type
        ).set(1)

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

        # Pre-initialize evaluation metrics so Grafana shows series
        # even before the first evaluation completes.
        for eval_type in self.enabled_eval_types:
            metrics.eval_win_rate.labels(eval_type=eval_type).set(0.0)
            metrics.eval_avg_game_length.labels(eval_type=eval_type).set(0.0)
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

        metrics.worker_state.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(WorkerState.WAITING)

        print(f"Worker {self.worker_id}: Starting evaluation loop...")
        print(f"Worker {self.worker_id}: Enabled eval types: {self.enabled_eval_types}")

        start_time = time.time()
        eval_count = 0
        last_check_version = 0

        # Track cumulative results per (version, eval_type) for this worker
        # Used for final aggregation when done with a version
        worker_cumulative: Dict[str, Dict[str, Any]] = {}

        while self.running:
            if num_iterations >= 0 and eval_count >= num_iterations:
                break

            # Check if there's a new model version
            current_version = self.current_model_version

            if current_version > last_check_version:
                # Queue eval tasks for the new version
                self._queue_eval_tasks(current_version)
                last_check_version = current_version

            # Find an eval type with work remaining
            batch_claimed = False
            for eval_type in self.enabled_eval_types:
                # Check global progress
                global_progress = self._get_eval_progress(current_version, eval_type)
                if global_progress >= self.eval_games:
                    continue  # Target reached for this eval type

                # Try to claim a batch
                batch_size = self._claim_eval_batch(current_version, eval_type)
                if batch_size is None or batch_size <= 0:
                    continue  # No batch available or target just reached

                batch_claimed = True
                task_key = f"v{current_version}:{eval_type}"

                print(
                    f"Worker {self.worker_id}: Running {eval_type} v{current_version} "
                    f"batch of {batch_size} games ({self.num_simulations} sims) "
                    f"[global: {global_progress}/{self.eval_games}]",
                    flush=True
                )
                self._refresh_metrics_registration(force=True)
                metrics.worker_phase.labels(
                    worker_id=self.worker_id, worker_type='eval'
                ).set(WorkerPhase.EVALUATING)
                metrics.worker_state.labels(
                    worker_id=self.worker_id, worker_type='eval'
                ).set(WorkerState.RUNNING)

                # Run the batch evaluation
                result = self._run_evaluation(current_version, eval_type, num_games=batch_size)

                if result is not None:
                    # Update metrics immediately (Prometheus aggregates across workers)
                    self._update_metrics(result, metrics)
                    eval_count += 1

                    if result.duration_seconds > 0:
                        metrics.worker_steps_per_second.labels(
                            worker_id=self.worker_id, worker_type='eval'
                        ).set(result.games_played / result.duration_seconds)

                    # Accumulate results for this worker
                    if task_key not in worker_cumulative:
                        worker_cumulative[task_key] = {
                            'games_played': 0, 'wins': 0, 'losses': 0, 'draws': 0,
                            'wins_single': 0, 'wins_gammon': 0, 'wins_backgammon': 0,
                            'losses_single': 0, 'losses_gammon': 0, 'losses_backgammon': 0,
                            'total_length': 0, 'total_points': 0.0, 'duration': 0.0,
                        }
                    cum = worker_cumulative[task_key]
                    cum['games_played'] += result.games_played
                    cum['wins'] += result.wins
                    cum['losses'] += result.losses
                    cum['draws'] += result.draws
                    cum['wins_single'] += result.wins_single
                    cum['wins_gammon'] += result.wins_gammon
                    cum['wins_backgammon'] += result.wins_backgammon
                    cum['losses_single'] += result.losses_single
                    cum['losses_gammon'] += result.losses_gammon
                    cum['losses_backgammon'] += result.losses_backgammon
                    cum['total_length'] += result.avg_game_length * result.games_played
                    cum['total_points'] += result.total_points
                    cum['duration'] += result.duration_seconds

                    # Check if global target reached after our batch
                    new_global_progress = self._get_eval_progress(current_version, eval_type)
                    is_final = new_global_progress >= self.eval_games

                    # Emit batch result (print + optional Redis store)
                    self._emit_batch_result(result, is_final=is_final)

                    # Log to MLflow after each batch
                    self._log_mlflow_eval_metrics(result)
                    self._refresh_metrics_registration(force=True)

                    # If this was the last batch for this eval type, clear cumulative
                    if is_final:
                        print(
                            f"Worker {self.worker_id}: Eval target reached for {eval_type} v{current_version} "
                            f"(this worker contributed {cum['games_played']} games)",
                            flush=True
                        )
                        del worker_cumulative[task_key]

                # After running a batch, break to re-check for new model versions
                break

            if not batch_claimed:
                # No work available for any eval type, wait and check for model updates
                self._check_and_update_model()
                metrics.worker_state.labels(
                    worker_id=self.worker_id, worker_type='eval'
                ).set(WorkerState.WAITING)
                metrics.worker_steps_per_second.labels(
                    worker_id=self.worker_id, worker_type='eval'
                ).set(0.0)
                time.sleep(5.0)

                # Refresh metrics registration
                self._refresh_metrics_registration(force=True)
                metrics.worker_phase.labels(
                    worker_id=self.worker_id, worker_type='eval'
                ).set(WorkerPhase.IDLE)

        # Mark worker as stopped
        metrics.worker_status.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(0)
        metrics.worker_phase.labels(
            worker_id=self.worker_id, worker_type='eval'
        ).set(WorkerPhase.IDLE)

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

        # Detailed point breakdown (gammon/backgammon)
        metrics.eval_wins_single.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.wins_single)

        metrics.eval_wins_gammon.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.wins_gammon)

        metrics.eval_wins_backgammon.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.wins_backgammon)

        metrics.eval_losses_single.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.losses_single)

        metrics.eval_losses_gammon.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.losses_gammon)

        metrics.eval_losses_backgammon.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).inc(result.losses_backgammon)

        metrics.eval_win_rate.labels(eval_type=result.eval_type).set(result.win_rate)

        metrics.eval_total_points.labels(eval_type=result.eval_type).set(result.total_points)

        metrics.eval_avg_points.labels(eval_type=result.eval_type).set(result.avg_points_won)

        metrics.eval_duration.labels(
            worker_id=self.worker_id, eval_type=result.eval_type
        ).observe(result.duration_seconds)

        metrics.eval_avg_game_length.labels(eval_type=result.eval_type).set(result.avg_game_length)

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
