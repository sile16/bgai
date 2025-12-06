import jax
import jax.numpy as jnp
import chex


def _scalar_equity_to_value_logits(equity: jnp.ndarray) -> jnp.ndarray:
    """Convert a scalar equity value to 6-way outcome logits.

    Maps equity in [-1, 1] to a 6-way distribution:
    [win, gammon_win, backgammon_win, loss, gammon_loss, backgammon_loss]

    For a heuristic evaluator like pip count, we don't have information about
    gammon/backgammon probabilities, so we put all probability mass on single wins/losses.

    Uses softmax-style logits where:
    - equity > 0: higher logit for win outcomes
    - equity < 0: higher logit for loss outcomes
    """
    # Scale equity to a reasonable logit range
    # Clamp to avoid extreme values
    equity = jnp.clip(equity, -0.99, 0.99)

    # Convert equity to logits
    # For positive equity: mostly predict single win
    # For negative equity: mostly predict single loss
    # Use temperature scaling to make distribution more peaked
    temperature = 0.5

    # Base logits: win outcomes get positive equity, loss outcomes get negative
    # Indices: [win, gammon_win, bg_win, loss, gammon_loss, bg_loss]
    win_logit = equity / temperature
    loss_logit = -equity / temperature

    # Create 6-way logits with most mass on single win/loss
    # Gammon/backgammon logits are set much lower since pip count can't predict them
    value_logits = jnp.array([
        win_logit,           # win (single)
        win_logit - 3.0,     # gammon win (unlikely based on pip count alone)
        win_logit - 5.0,     # backgammon win (very unlikely)
        loss_logit,          # loss (single)
        loss_logit - 3.0,    # gammon loss (unlikely)
        loss_logit - 5.0,    # backgammon loss (very unlikely)
    ])

    return value_logits


@jax.jit
def backgammon_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey):
    """Calculates value based on pip count difference. Returns 6-way value logits.

    Ignores params/key. The board is always from the current player's perspective,
    current player is positive numbers, opponent is negative.

    Returns:
        Tuple of (policy_logits, value_logits) where value_logits is shape (6,)
        representing outcome distribution over:
        [win, gammon_win, backgammon_win, loss, gammon_loss, backgammon_loss]
    """
    board = state._board
    pips = state._board[1:25]

    # Calculate pip counts for current player and opponent
    current_pips = jnp.sum(jnp.maximum(0, pips) * jnp.arange(1, 25, dtype=jnp.int32))
    opponent_pips = jnp.sum(jnp.maximum(0, -pips) * jnp.arange(1, 25, dtype=jnp.int32))

    # Add born-off checkers with appropriate weights
    # Using 25 points for born-off checkers (standard backgammon pip count)
    current_born_off = board[26] * 25  # Current player's born-off checkers
    opponent_born_off = board[27] * 25  # Opponent's born-off checkers

    # Calculate total pips for normalization
    total_pips = current_pips + opponent_pips + current_born_off + opponent_born_off + 1e-6

    # Calculate normalized equity value between -1 and 1
    # Positive value means current player is ahead
    equity = (opponent_pips + opponent_born_off - current_pips - current_born_off) / total_pips

    # Ensure stochastic states are not evaluated directly
    equity = jnp.where(state._is_stochastic, 0.0, equity)

    # Convert equity to 6-way value logits
    value_logits = _scalar_equity_to_value_logits(equity)

    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)

    return policy_logits, value_logits