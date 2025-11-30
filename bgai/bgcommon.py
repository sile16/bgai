import jax
import chex
from typing import Tuple
from pgx import backgammon as bg
from pgx.core import StepMetadata

def backgammon_step_fn(state: bg.State, action: int, key: chex.PRNGKey) -> Tuple[bg.State, StepMetadata]:
    """Combined step function for backgammon environment that handles both deterministic and stochastic actions."""
    # print(f"[DEBUG-BG_STEP-{time.time()}] Called with state (stochastic={state._is_stochastic}), action={action}") # Optional debug

    # Handle stochastic vs deterministic branches
    def stochastic_branch(operand):
        s, a, _ = operand # state, action, key (key ignored for stochastic step)
        # Use env instance captured by closure (assuming env is accessible in this scope)
        return env.stochastic_step(s, a)

    def deterministic_branch(operand):
        s, a, k = operand # state, action, key
        # Use env instance captured by closure
        return env.step(s, a, k)

    # Use conditional to route to the appropriate branch
    # The key is only needed for the deterministic branch
    new_state = jax.lax.cond(
        state._is_stochastic,
        stochastic_branch,
        deterministic_branch,
        (state, action, key) # Pass all required operands
    )

    # Create standard metadata
    metadata = StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count
    )

    return new_state, metadata


