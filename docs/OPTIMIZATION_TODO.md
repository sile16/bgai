# BGAI Optimization TODO

## Priority: Get Distributed System Working First
Before pursuing further optimizations, ensure the distributed training system is functioning correctly.

---

## Future Optimizations

### 1. MCTS Tree Caching (High Impact)
**Observation:** On each new epoch, the first N MCTS iterations from the starting position are identical across all games.

**Proposed Solution:**
- Build a deep "root cache" tree once at epoch start (e.g., 5000 iterations)
- Clone this cached tree for each new game
- Continue MCTS from the cached state rather than from scratch

**Benefits:**
- Amortize initial tree-building cost across all games
- Better exploration of opening positions
- Could provide significant speedup (TBD - needs benchmarking)

**Considerations:**
- Cache invalidates when NN weights update (once per epoch)
- Memory tradeoff: larger cached tree vs. recomputation savings
- Need efficient tree cloning mechanism (JAX pytrees should work)

### 2. Batched NN Inference Across Games
**Current State:** Each MCTS tree evaluates NN one state at a time, even with vmapped games.

**Proposed Solution:**
- Collect leaf node observations from all 128 games
- Batch them into single NN forward pass
- Distribute results back to respective trees

**Challenges:**
- Requires modifying MCTS internals
- Synchronization across games at different tree depths
- May conflict with vmap parallelism

### 3. GPU Hybrid for Larger Networks
**Current State:** On Mac M1, hybrid CPU+GPU shows minimal benefit (~8%) due to:
- Transfer overhead for single observations
- MCTS tree operations dominate, not NN inference

**When to Revisit:**
- If using much larger neural networks
- If batched NN inference (optimization #2) is implemented
- On CUDA systems (no Metal lax.cond bug)

---

## Benchmark Results Summary (Mac M1 Pro)

### Vmapped Games (CPU-only)
| Batch Size | Steps/sec | Speedup |
|------------|-----------|---------|
| 1 | 0.2 | 1x |
| 16 | 2.5 | 12.5x |
| 128 | 17.8 | **115x** |

### Raw NN Inference (GPU Metal)
| Batch Size | Inferences/sec |
|------------|----------------|
| 1 | 507 |
| 128 | 56,986 (112x) |

### Recommendation for Mac
Use CPU-only with vmap batching (batch_size=128):
- Simpler code (no pure_callback workarounds)
- Avoids JAX Metal bugs
- 115x improvement already achieved

---

## Technical Notes

### JAX Metal Bug Workaround
The Metal backend has a serialization bug with `lax.cond` returning boolean values.
Workaround: Use `jax.pure_callback` to isolate GPU ops from JAX tracing.
See: `benchmarks/bench_hybrid_*.py`

### Turbozero Batching Pattern
Turbozero uses `jax.vmap(collect_steps, ...)` to run multiple games in parallel.
The MCTS is designed to be vmap-compatible.
See: `core/training/train.py:689`
