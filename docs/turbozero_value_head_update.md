TurboZero value head update (6‑way outcomes)
============================================

Context
-------
- TurboZero now outputs a 6-way outcome head: [win, gammon win, backgammon win, loss, gammon loss, backgammon loss].
- MCTS expects eval_fns to return `(policy_logits, value_probs[6])`. `terminal_value_probs_from_reward` maps signed point rewards (+/-1/2/3) to the 6-way target; no rescaling to [-1, 1] is needed.
- `probs_to_equity` converts 6-way probs to equity in [0, 1]; `get_value` returns that equity.

Required code changes
---------------------
1) Models / heads
   - Switch all value heads from size 1 to size 6 and remove `squeeze` on the value axis:
     - `distributed/workers/game_worker.py` (ResNetTurboZero)
     - `distributed/workers/eval_worker.py` (ResNetTurboZero)
     - Benchmarks: `bench_two_player.py`, `bench_training_loop.py`, `bench_hybrid_*`, `bench_model_comparison.py`, `bench_vmap_games.py`, `bench_batched_cpu.py` (search `Dense(1)` and `value_head_out_size=1`).
     - Heuristic eval: `bgai/bgevaluators.py`’s `backgammon_pip_count_eval` must output 6-way outcome logits/probs (see heuristic mapping below).

2) Losses
   - `distributed/workers/training_worker.py`:
     - Update `weighted_az_loss_fn` to mirror TurboZero’s new loss: value term is cross-entropy on value logits vs `terminal_value_probs_from_reward(reward_for_cur_player)`. Remove scalar MSE value loss.
     - Ensure metrics in that loss use value CE/entropy/top-1, not absolute error vs scalar.
   - Keep `self._loss_fn = az_default_loss_fn` if it comes from the updated TurboZero (already supports 6-way).

3) Bearoff handling
   - `_apply_bearoff_values_to_batch` can keep writing scalar rewards (points). The value loss must use `terminal_value_probs_from_reward`, so no change needed here beyond the loss update.
   - `bearoff_value_weight` should weight the new value CE term.

4) Metrics/logging
   - `training_worker` bearoff/non-bearoff metrics block still assumes scalar `pred_value`. Rework to:
     - Convert value logits -> probs; compute per-sample value CE; compute bearoff/non-bearoff averages.
     - Optionally log equity error by converting probs to equity.
   - MLflow/prometheus metrics: `value_loss` now means value CE; update log message text to avoid confusion. Legacy scalar value metrics can be dropped unless needed.

5) Evaluator outputs and surprise scoring
   - `GameWorker` uses `self._evaluator.get_value(...)` (now equity in [0, 1]) for surprise scoring vs `final_rewards` (points). Decide if you want to compare equity vs point outcome or convert both to the same scale; current logic may need a recheck but not a blocker for compilation.

6) Heuristic 6-way mapping
   - Convert scalar pip-based heuristic to 6-way:
     - Deterministic: map a scalar “equity guess” to sign (+/-) and magnitude bucket (1/2/3), emit one-hot over the 6 outcomes (win/gammon/backgammon vs loss variants).
     - Smoother option: spread mass across the three magnitude buckets with a softmax on |equity|*scale, then apply sign to pick the win vs loss side.

7) Benchmarks/training demos
   - After updating heads to 6, adjust any manual loss snippets (e.g., `bench_training_loop.py` around value loss) to use value CE with `terminal_value_probs_from_reward` and remove `squeeze` assumptions.

8) Notebooks/docs
   - Update any code snippets mentioning `value_head_out_size=1` to 6 and note the new loss formulation.

Checks after implementation
---------------------------
- Run unit/benchmarks that cover MCTS eval paths to ensure shapes match (policy logits, value_probs[6]).
- Validate training step runs end-to-end with Redis samples and bearoff weighting enabled.
- Spot-check MLflow metrics to confirm value_loss corresponds to CE and no shape errors occur.
