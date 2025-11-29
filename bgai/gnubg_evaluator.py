from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import chex
import gnubg
import jax
import jax.numpy as jnp
import numpy as np
from core.evaluators.evaluator import EvalOutput, Evaluator
from pgx import backgammon as bg

# The bridge helpers live in the sibling pgx repo (../pgx/tools).
# Import them directly if pgx is on the path; otherwise, add the local clone.
try:  # pragma: no cover - import convenience logic
    from tools.gnubg_bridge import (  # type: ignore
        enumerate_pgx_moves,
        full_move_to_steps,
        pgx_to_gnubg_board,
    )
except ModuleNotFoundError:  # pragma: no cover - import convenience logic
    pgx_repo = Path(__file__).resolve().parents[2] / "pgx"
    if str(pgx_repo) not in sys.path:
        sys.path.append(str(pgx_repo))
    from tools.gnubg_bridge import (  # type: ignore
        enumerate_pgx_moves,
        full_move_to_steps,
        pgx_to_gnubg_board,
    )


def _gnubg_select_action_host(board: np.ndarray, dice: np.ndarray, legal_action_mask: np.ndarray) -> np.ndarray:
    """Host-side function to select an action using gnubg.

    This runs outside JAX tracing and can use arbitrary Python code.
    Returns the selected action as an int32 array.

    Note: In backgammon, each turn may involve multiple moves (one per die).
    The legal_action_mask tells us which actions are valid in the current
    partial-turn state. We must respect this mask even when gnubg suggests
    a full move sequence.
    """
    # Convert to numpy if needed (JAX arrays are passed as numpy by pure_callback)
    board = np.asarray(board, dtype=np.int32)
    dice = np.asarray(dice, dtype=np.int32)
    legal_action_mask = np.asarray(legal_action_mask, dtype=bool)

    dice_tuple = (int(dice[0]) + 1, int(dice[1]) + 1)

    legal_sequences: Dict[Tuple[Tuple[int, int], ...], Sequence[int]] = enumerate_pgx_moves(board, dice_tuple)

    # Try to get gnubg's best move
    best_action: int | None = None
    try:
        gnubg_board = pgx_to_gnubg_board(board)
        best_move = gnubg.best_move(gnubg_board, dice_tuple[0], dice_tuple[1])
        best_steps = full_move_to_steps(best_move)
        if best_steps in legal_sequences:
            seq_actions = legal_sequences[best_steps]
            # Find the first action in the sequence that is currently legal
            for action in seq_actions:
                if action < len(legal_action_mask) and legal_action_mask[action]:
                    best_action = action
                    break
    except Exception:
        pass

    if best_action is not None:
        return np.array(best_action, dtype=np.int32)

    # Fallback: find any legal action from any sequence gnubg might suggest
    for steps, seq_actions in legal_sequences.items():
        for action in seq_actions:
            if action < len(legal_action_mask) and legal_action_mask[action]:
                return np.array(action, dtype=np.int32)

    # Ultimate fallback: first legal action
    legal_indices = np.nonzero(legal_action_mask)[0]
    if legal_indices.size:
        return np.array(legal_indices[0], dtype=np.int32)
    return np.array(0, dtype=np.int32)


class GnubgEvaluator(Evaluator):
    """
    TurboZero-compatible evaluator that delegates move selection to GNU Backgammon.

    - Stochastic states (dice selection) are sampled from pgx's stochastic_action_probs.
    - Deterministic states ask gnubg for its best move, then map that move back into the
      pgx action space via the bridge helpers in ../pgx/tools/gnubg_bridge.py.

    This evaluator uses jax.pure_callback to call gnubg from within JAX-traced code,
    making it compatible with jax.vmap and jax.jit.

    This is intended for head-to-head evaluation of a learned policy against gnubg.
    """

    def __init__(self, env: bg.Backgammon | None = None):
        super().__init__(discount=-1.0)
        self.env = env or bg.Backgammon()
        self._stochastic_probs = jnp.asarray(self.env.stochastic_action_probs, dtype=jnp.float32)
        self._num_stochastic_actions = int(self._stochastic_probs.shape[0])
        self._num_actions = self.env.num_actions

    def init(self, *args, **kwargs) -> chex.Array:
        # Stateful tracking is unnecessary; return a scalar placeholder for tree compatibility.
        return jnp.array(0, dtype=jnp.int32)

    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:  # pylint: disable=unused-argument
        return self.init()

    def get_value(self, state: chex.ArrayTree) -> chex.Array:  # pylint: disable=unused-argument
        # gnubg provides moves, not value estimates; return a neutral placeholder.
        return jnp.array(0.0, dtype=jnp.float32)

    def step(self, state: chex.ArrayTree, action: chex.Array) -> chex.ArrayTree:  # pylint: disable=unused-argument
        # No internal state to update between steps.
        return state

    def evaluate(  # pylint: disable=arguments-differ
        self,
        key: chex.PRNGKey,
        eval_state: chex.ArrayTree,
        env_state: chex.ArrayTree,
        root_metadata=None,
        params=None,
        env_step_fn=None,
        **kwargs,
    ) -> EvalOutput:
        del params, env_step_fn, root_metadata, kwargs  # Unused but kept for interface parity.

        # Use jax.lax.cond for JAX-compatible branching on stochastic state
        is_stochastic = env_state._is_stochastic

        # For stochastic states: sample dice roll
        stochastic_action = jax.random.choice(
            key, self._num_stochastic_actions, p=self._stochastic_probs
        )
        stochastic_policy = self._one_hot_jax(self._num_stochastic_actions, stochastic_action)

        # For deterministic states: call gnubg via pure_callback
        # Pass JAX arrays directly - they will be converted to numpy in the callback
        deterministic_action = jax.pure_callback(
            _gnubg_select_action_host,
            jax.ShapeDtypeStruct((), jnp.int32),  # result_shape_dtypes
            env_state._board,
            env_state._dice,
            env_state.legal_action_mask,
            vmap_method='sequential',  # Each call is independent, run sequentially
        )
        deterministic_policy = self._one_hot_jax(
            self._num_actions, deterministic_action, env_state.legal_action_mask
        )

        # Select based on stochastic flag using jax.lax.cond-compatible logic
        action = jnp.where(is_stochastic, stochastic_action, deterministic_action)
        policy_weights = jnp.where(
            is_stochastic,
            jnp.pad(stochastic_policy, (0, self._num_actions - self._num_stochastic_actions), constant_values=-jnp.inf),
            deterministic_policy,
        )

        return EvalOutput(eval_state=eval_state, action=action, policy_weights=policy_weights)

    @staticmethod
    def _one_hot_jax(size: int, index: chex.Array, legal_mask: chex.Array | None = None) -> jnp.ndarray:
        """Return sparse logits with a single preferred action (JAX-compatible)."""
        logits = jnp.full((size,), -jnp.inf, dtype=jnp.float32)
        # Clamp index to valid range for safe indexing
        safe_index = jnp.clip(index, 0, size - 1)
        logits = logits.at[safe_index].set(0.0)
        if legal_mask is not None:
            mask_arr = jnp.asarray(legal_mask, dtype=bool)
            logits = jnp.where(mask_arr[:size], logits, -jnp.inf)
        return logits
