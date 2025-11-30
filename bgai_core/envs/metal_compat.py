"""Metal-compatible wrappers for PGX environments.

This module provides wrappers that work around a known JAX Metal bug where
jax.lax.cond fails when branches return boolean values. The bug manifests as:
    XlaRuntimeError: INTERNAL: Unable to serialize MPS module

See: https://github.com/jax-ml/jax/issues/20401

The issue is tracked in JAX but remains unfixed as of jax-metal 0.1.1 (Oct 2024).
The bug affects JAX versions 0.4.25+ with jax-metal 0.0.6+.

## Solution Options

1. **Modify PGX directly** (recommended for forks): Replace `lax.cond` with
   `jnp.where` in core.py and backgammon.py. See docs/pgx_metal_fix_proposal.md.

2. **Use CPU backend**: Actually faster for backgammon anyway.
   Set `JAX_PLATFORMS=cpu` before importing JAX.

3. **Use these wrappers**: For when you can't modify PGX.

Usage:
    from core.envs.metal_compat import BackgammonMetalEnv

    env = BackgammonMetalEnv(short_game=True)
    state = env.step(state, action, key)  # Works on Metal
"""

from typing import Optional, TypeVar
import jax
import jax.numpy as jnp

StateT = TypeVar('StateT')

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def is_metal_backend() -> bool:
    """Check if JAX is using the Metal backend."""
    try:
        return jax.default_backend().upper() == 'METAL'
    except Exception:
        return False


def force_cpu_on_metal():
    """Force CPU backend if on Apple Silicon.

    MUST be called before importing JAX.
    Actually provides better performance for backgammon anyway.

    Returns:
        True if CPU was forced, False otherwise.
    """
    import os
    import platform

    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        if 'JAX_PLATFORMS' not in os.environ:
            os.environ['JAX_PLATFORMS'] = 'cpu'
            return True
    return False


class BackgammonMetalEnv:
    """Metal-compatible wrapper for PGX Backgammon.

    This wrapper reimplements step() using jnp.where instead of lax.cond
    to work around the JAX Metal serialization bug.

    Note: This wrapper is designed for the sile16/pgx fork which may have
    different internal APIs than upstream PGX. The wrapper intercepts step()
    and provides Metal-compatible logic.

    Example:
        >>> from core.envs.metal_compat import BackgammonMetalEnv
        >>>
        >>> env = BackgammonMetalEnv(short_game=True)
        >>> key = jax.random.PRNGKey(42)
        >>> state = env.init(key)
        >>>
        >>> # Handle stochastic step (dice roll)
        >>> if state._is_stochastic:  # or state.is_stochastic for upstream
        >>>     state = env.stochastic_step(state, dice_action)
        >>> else:
        >>>     state = env.step(state, action, key)
    """

    def __init__(self, env=None, **kwargs):
        """Initialize the wrapper.

        Args:
            env: Optional existing Backgammon environment to wrap.
                 If None, creates a new one with the provided kwargs.
            **kwargs: Arguments passed to Backgammon() if env is None.
        """
        if env is None:
            from pgx.backgammon import Backgammon
            self._env = Backgammon(**kwargs)
        else:
            self._env = env

        # Detect which field name is used for stochastic flag
        key = jax.random.PRNGKey(0)
        sample_state = self._env.init(key)
        self._stochastic_field = '_is_stochastic' if hasattr(sample_state, '_is_stochastic') else 'is_stochastic'

    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self._env, name)

    def init(self, key):
        """Initialize the environment."""
        return self._env.init(key)

    def observe(self, state, player_id=None):
        """Get observation."""
        return self._env.observe(state, player_id)

    def stochastic_step(self, state, action):
        """Stochastic step (dice roll) - delegates to wrapped env."""
        return self._env.stochastic_step(state, action)

    def set_dice(self, state, dice):
        """Set dice - delegates to wrapped env."""
        return self._env.set_dice(state, dice)

    def step(self, state, action, key=None):
        """Metal-compatible step function.

        This reimplements the step logic using jnp.where instead of lax.cond.
        """
        if key is None:
            raise ValueError("key is required for backgammon step")

        return self._step_metal(state, action, key)

    def _step_metal(self, state, action, key):
        """Core Metal-compatible step implementation."""
        # Import internal functions from backgammon module
        from pgx.backgammon import (
            _move, _update_playable_dice, _is_all_off,
            _calc_win_score, _is_turn_end, _flip_board,
            _set_playable_dice, _roll_dice, TRUE, FALSE,
        )

        # Try to import _legal_action_mask with correct signature
        from pgx import backgammon as bg_module
        import inspect
        _legal_action_mask = bg_module._legal_action_mask
        sig = inspect.signature(_legal_action_mask)
        num_params = len(sig.parameters)

        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player
        already_done = state.terminated | state.truncated

        # =========================================
        # Metal-compatible _update_by_action logic
        # =========================================
        is_no_op = action // 6 == 0
        board = _move(state._board, action)
        played_dice_num = jnp.int32(state._played_dice_num + 1)
        playable_dice = _update_playable_dice(
            state._playable_dice, state._played_dice_num, state._dice, action
        )

        # Handle different _legal_action_mask signatures
        if num_params == 4:
            # Fork version: (board, playable_dice, turn_dice, played_dice_num)
            legal_mask = _legal_action_mask(board, playable_dice, state._dice, played_dice_num)
        else:
            # Upstream version: (board, playable_dice)
            legal_mask = _legal_action_mask(board, playable_dice)

        # jnp.where instead of lax.cond for no-op handling
        updated_board = jnp.where(is_no_op, state._board, board)
        updated_playable = jnp.where(is_no_op, state._playable_dice, playable_dice)
        updated_played_num = jnp.where(is_no_op, state._played_dice_num, played_dice_num)
        updated_legal_mask = jnp.where(is_no_op, state.legal_action_mask, legal_mask)

        updated_state = state.replace(
            _board=updated_board,
            _playable_dice=updated_playable,
            _played_dice_num=updated_played_num,
            legal_action_mask=updated_legal_mask,
            _step_count=state._step_count + 1,
        )

        # =========================================
        # Metal-compatible win check
        # =========================================
        is_all_off = _is_all_off(updated_state._board)

        # Winning state
        win_score = _calc_win_score(updated_state._board)
        winner = updated_state.current_player
        loser = 1 - winner
        win_reward = jnp.ones_like(updated_state.rewards)
        win_reward = win_reward.at[winner].set(win_score)
        win_reward = win_reward.at[loser].set(-win_score)

        # =========================================
        # Metal-compatible turn change
        # =========================================
        should_change = _is_turn_end(updated_state) | is_no_op

        # Compute turn change state
        new_board = _flip_board(updated_state._board)
        new_turn = (updated_state._turn + 1) % 2
        new_player = (updated_state.current_player + 1) % 2
        new_dice = _roll_dice(key)
        new_playable = _set_playable_dice(new_dice)
        new_played_num = jnp.int32(0)

        if num_params == 4:
            new_legal_mask = _legal_action_mask(new_board, new_playable, new_dice, new_played_num)
        else:
            new_legal_mask = _legal_action_mask(new_board, new_playable)

        # Select between changed/unchanged using jnp.where
        final_board = jnp.where(should_change, new_board, updated_state._board)
        final_turn = jnp.where(should_change, new_turn, updated_state._turn)
        final_player = jnp.where(should_change, new_player, updated_state.current_player)
        final_dice = jnp.where(should_change, new_dice, updated_state._dice)
        final_playable = jnp.where(should_change, new_playable, updated_state._playable_dice)
        final_played_num = jnp.where(should_change, new_played_num, updated_state._played_dice_num)
        final_legal = jnp.where(should_change, new_legal_mask, updated_state.legal_action_mask)
        final_stochastic = jnp.where(should_change, TRUE, getattr(updated_state, self._stochastic_field))

        # Select between win/no-win using jnp.where
        final_rewards = jnp.where(is_all_off, win_reward, updated_state.rewards)
        final_terminated = jnp.where(is_all_off, TRUE, updated_state.terminated)

        # For non-winning case, use the turn-change results
        final_player = jnp.where(is_all_off, updated_state.current_player, final_player)
        final_board = jnp.where(is_all_off, updated_state._board, final_board)
        final_dice = jnp.where(is_all_off, updated_state._dice, final_dice)
        final_playable = jnp.where(is_all_off, updated_state._playable_dice, final_playable)
        final_played_num = jnp.where(is_all_off, updated_state._played_dice_num, final_played_num)
        final_turn = jnp.where(is_all_off, updated_state._turn, final_turn)
        final_legal = jnp.where(is_all_off, updated_state.legal_action_mask, final_legal)
        final_stochastic = jnp.where(is_all_off, getattr(updated_state, self._stochastic_field), final_stochastic)

        # =========================================
        # Metal-compatible already_done handling
        # =========================================
        final_rewards = jnp.where(already_done, jnp.zeros_like(state.rewards), final_rewards)
        final_terminated = jnp.where(already_done, state.terminated, final_terminated)
        final_player = jnp.where(already_done, state.current_player, final_player)
        final_board = jnp.where(already_done, state._board, final_board)
        final_dice = jnp.where(already_done, state._dice, final_dice)
        final_playable = jnp.where(already_done, state._playable_dice, final_playable)
        final_played_num = jnp.where(already_done, state._played_dice_num, final_played_num)
        final_turn = jnp.where(already_done, state._turn, final_turn)
        final_legal = jnp.where(already_done, state.legal_action_mask, final_legal)
        final_stochastic = jnp.where(already_done, getattr(state, self._stochastic_field), final_stochastic)
        final_step_count = jnp.where(already_done, state._step_count, updated_state._step_count)

        # =========================================
        # Metal-compatible illegal action handling
        # =========================================
        penalty = self._env._illegal_action_penalty
        illegal_reward = jnp.ones_like(final_rewards) * (-1 * penalty) * (self._env.num_players - 1)
        illegal_reward = illegal_reward.at[current_player].set(penalty)

        final_rewards = jnp.where(is_illegal, illegal_reward, final_rewards)
        final_terminated = jnp.where(is_illegal, TRUE, final_terminated)

        # Set all legal when terminated
        final_legal = jnp.where(
            final_terminated,
            jnp.ones_like(final_legal),
            final_legal
        )

        # Build final state with appropriate field name
        replace_kwargs = {
            'rewards': final_rewards,
            'terminated': final_terminated,
            'current_player': final_player,
            '_board': final_board,
            '_dice': final_dice,
            '_playable_dice': final_playable,
            '_played_dice_num': final_played_num,
            '_turn': final_turn,
            'legal_action_mask': final_legal,
            '_step_count': final_step_count,
            self._stochastic_field: final_stochastic,
        }

        final_state = updated_state.replace(**replace_kwargs)

        observation = self._env.observe(final_state)
        return final_state.replace(observation=observation)

    @property
    def stochastic_action_probs(self):
        return self._env.stochastic_action_probs

    @property
    def num_actions(self):
        return self._env.num_actions

    @property
    def num_players(self):
        return self._env.num_players

    @property
    def observation_shape(self):
        return self._env.observation_shape

    @property
    def id(self):
        return self._env.id

    @property
    def version(self):
        return self._env.version
