import jax
import jax.numpy as jnp
import chex


@jax.jit
def backgammon_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey):
    """Calculates value based on pip count difference. Ignores params/key.
    The board is always from the current players perspective, 
    current player is positive numbers opponent is negative."""
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
    
    # Calculate normalized value between -1 and 1
    # Positive value means current player is ahead
    value = (opponent_pips + opponent_born_off - current_pips - current_born_off) / total_pips
    
    # Ensure stochastic states are not evaluated directly
    value = jnp.where(state._is_stochastic, jnp.nan, value)
    
    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    
    return policy_logits, jnp.array(value)