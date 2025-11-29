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


def _is_state_stochastic(state: chex.ArrayTree) -> bool:
    """Handle both pgx._is_stochastic and a potential is_stochastic alias."""
    if hasattr(state, "is_stochastic"):
        return bool(getattr(state, "is_stochastic"))
    if hasattr(state, "_is_stochastic"):
        return bool(getattr(state, "_is_stochastic"))
    return False


class GnubgEvaluator(Evaluator):
    """
    TurboZero-compatible evaluator that delegates move selection to GNU Backgammon.

    - Stochastic states (dice selection) are sampled from pgx's stochastic_action_probs.
    - Deterministic states ask gnubg for its best move, then map that move back into the
      pgx action space via the bridge helpers in ../pgx/tools/gnubg_bridge.py.

    This is intended for head-to-head evaluation of a learned policy against gnubg.
    """

    def __init__(self, env: bg.Backgammon | None = None):
        super().__init__(discount=-1.0)
        self.env = env or bg.Backgammon()
        self._stochastic_probs = np.asarray(self.env.stochastic_action_probs, dtype=np.float32)
        self._num_stochastic_actions = int(self._stochastic_probs.shape[0])

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

        if _is_state_stochastic(env_state):
            # Sample a dice roll action using pgx's prescribed distribution.
            action = int(jax.random.choice(key, self._num_stochastic_actions, p=jnp.array(self._stochastic_probs)))
            policy_weights = self._one_hot(self._num_stochastic_actions, action)
            return EvalOutput(eval_state=eval_state, action=action, policy_weights=policy_weights)

        # Deterministic step: ask gnubg for a move, then map it to pgx action ids.
        board = np.asarray(jax.device_get(env_state._board), dtype=np.int32)  # type: ignore[attr-defined]
        dice_raw = np.asarray(jax.device_get(env_state._dice), dtype=np.int32)  # type: ignore[attr-defined]
        dice = (int(dice_raw[0]) + 1, int(dice_raw[1]) + 1)

        legal_sequences: Dict[Tuple[Tuple[int, int], ...], Sequence[int]] = enumerate_pgx_moves(board, dice)
        chosen_steps = self._select_gnubg_steps(board, dice, legal_sequences)
        seq_actions = legal_sequences.get(chosen_steps, [])

        if seq_actions:
            action = int(seq_actions[0])
        else:
            action = self._fallback_action(env_state)

        policy_weights = self._one_hot(
            size=self.env.num_actions,
            index=action,
            legal_mask=np.asarray(env_state.legal_action_mask),
        )
        return EvalOutput(eval_state=eval_state, action=action, policy_weights=policy_weights)

    def _select_gnubg_steps(
        self,
        board: np.ndarray,
        dice: Tuple[int, int],
        legal_sequences: Dict[Tuple[Tuple[int, int], ...], Sequence[int]],
    ) -> Tuple[Tuple[int, int], ...]:
        """Ask gnubg for its best move and ensure it lines up with pgx's move set."""
        try:
            gnubg_board = pgx_to_gnubg_board(board)
            best_move = gnubg.best_move(gnubg_board, dice[0], dice[1])
            best_steps = full_move_to_steps(best_move)
        except Exception:
            best_steps = None

        if best_steps in legal_sequences:
            return best_steps  # type: ignore[return-value]

        # Fallback: use the first legal pgx sequence to keep play moving.
        if legal_sequences:
            return next(iter(legal_sequences.keys()))
        return tuple()

    @staticmethod
    def _one_hot(size: int, index: int, legal_mask: Sequence[bool] | None = None) -> jnp.ndarray:
        """Return sparse logits with a single preferred action."""
        logits = jnp.full((size,), -jnp.inf, dtype=jnp.float32)
        if 0 <= index < size:
            logits = logits.at[index].set(0.0)
        if legal_mask is not None:
            mask_arr = jnp.asarray(legal_mask, dtype=bool)
            logits = jnp.where(mask_arr, logits, -jnp.inf)
        return logits

    @staticmethod
    def _fallback_action(env_state: chex.ArrayTree) -> int:
        """Pick the first legal action (including no-ops) when gnubg mapping fails."""
        mask = np.asarray(env_state.legal_action_mask)
        legal_indices = np.nonzero(mask)[0]
        if legal_indices.size:
            return int(legal_indices[0])
        return 0
