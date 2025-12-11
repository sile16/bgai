# BGAI Architecture Refactoring Plan

## Overview

Remove Ray completely and implement a simpler, more reliable Redis-only distributed architecture with MLflow for experiment tracking.

## Current Architecture (Ray-based)

```
HEAD NODE                          REMOTE WORKERS
┌──────────────────────┐          ┌──────────────────┐
│ Ray Head             │          │ Ray Worker Node  │
│ ├─ Coordinator Actor │◄────────►│ ├─ GameWorker    │
│ ├─ TrainingWorker    │  (Ray)   │ └─ EvalWorker    │
│ └─ Redis Buffer      │          └──────────────────┘
└──────────────────────┘                   ▲
                                           │ UNSTABLE
                                           │ (raylet crashes)
```

**Problems:**
- Ray multi-node clusters unstable on Mac
- Complex actor lifecycle management
- Heavy dependency (~500MB)
- Difficult to debug

## New Architecture (Redis-only)

```
HEAD NODE                              REMOTE WORKERS
┌─────────────────────────────────┐   ┌────────────────────────┐
│ ┌───────────┐  ┌─────────────┐  │   │  Game Worker (Python)  │
│ │Coordinator│  │   MLflow    │  │   │  ├─ Prometheus :9100   │
│ │ (Python)  │  │   :5000     │  │   │  └─ Redis client       │
│ └─────┬─────┘  └─────────────┘  │   └───────────┬────────────┘
│       │                          │               │
│ ┌─────▼─────────────────────┐   │               │
│ │         Redis             │◄──┼───────────────┘
│ │  ├─ Model weights         │   │
│ │  ├─ Worker heartbeats     │   │   ┌────────────────────────┐
│ │  ├─ Replay buffer         │   │   │  Eval Worker (Python)  │
│ │  └─ Cluster state         │   │   │  ├─ Prometheus :9200   │
│ └───────────────────────────┘   │   │  └─ Redis client       │
│                                  │   └───────────┬────────────┘
│ ┌───────────────────────────┐   │               │
│ │ Prometheus + Grafana      │◄──┼───────────────┘
│ │  ├─ Node metrics          │   │
│ │  └─ Training metrics      │   │
│ └───────────────────────────┘   │
│                                  │
│ ┌───────────────────────────┐   │
│ │   Training Worker         │   │
│ │   └─ MLflow logging       │   │
│ └───────────────────────────┘   │
└─────────────────────────────────┘
```

## Redis Key Schema

```
# Cluster State
bgai:model:version          = int (current model version)
bgai:model:weights          = bytes (pickled model params)
bgai:model:config           = json (training config hash)
bgai:run:id                 = string (current MLflow run ID)
bgai:run:status             = string (running|paused|stopped)

# Worker Registry (with TTL for auto-cleanup)
bgai:workers:{worker_id}    = json {
    type: "game"|"training"|"eval",
    device_type: "cuda"|"metal"|"cpu",
    device_name: "RTX 4090",
    hostname: "gpu-server",
    metrics_port: 9100,
    status: "working"|"idle",
    games_generated: 1234,
    model_version: 42,
    last_heartbeat: timestamp
}  TTL=60s (refreshed by heartbeat)

# Replay Buffer (existing)
bgai:buffer:episodes        = list of episode keys
bgai:buffer:episode:{id}    = bytes (serialized episode)
bgai:buffer:stats           = json {size, oldest, newest}

# Metrics Discovery
bgai:metrics:targets        = json [{host, port, labels}...]
```

## MLflow Integration

**Tracking:**
- Run name: `{config_hash}_{timestamp}`
- Parameters: All config values from distributed.yaml
- Metrics: loss, policy_loss, value_loss, eval_win_rate, games_collected
- Artifacts: Config file, model checkpoints, training curves

**Training Run Management:**
```python
# Start new run
mlflow.start_run(run_name="exp_001")
mlflow.log_params(config)

# Resume from checkpoint
mlflow.start_run(run_id=existing_run_id)

# Clean old runs
mlflow.delete_run(run_id)
```

**CLI Commands:**
```bash
# List training runs
python -m distributed.cli.main runs list

# Start new training run
python -m distributed.cli.main runs start --config configs/distributed.yaml

# Resume training run
python -m distributed.cli.main runs resume --run-id abc123

# Delete training run
python -m distributed.cli.main runs delete --run-id abc123 --delete-checkpoints
```

## File Changes

### Delete (Ray-specific)
- No files deleted, but Ray code removed from existing files

### Modify

1. **`distributed/coordinator/head_node.py`** (major rewrite)
   - Remove `@ray.remote` decorator
   - Change to standalone Python class
   - Use Redis for all state storage
   - Add MLflow run management
   - Add training run lifecycle (start/pause/resume/stop)

2. **`distributed/workers/base_worker.py`** (major rewrite)
   - Remove Ray dependencies
   - Redis-based heartbeat (set key with TTL)
   - Redis-based model weight fetching
   - Standalone process lifecycle

3. **`distributed/workers/game_worker.py`** (moderate changes)
   - Remove `@ray.remote`
   - Update to use new base_worker
   - Add system metrics collection (CPU, memory, GPU)

4. **`distributed/workers/training_worker.py`** (moderate changes)
   - Remove `@ray.remote`
   - Add MLflow logging
   - Add checkpoint resume logic

5. **`distributed/workers/eval_worker.py`** (moderate changes)
   - Remove `@ray.remote`
   - Update to use new base_worker
   - Log eval results to MLflow

6. **`distributed/cli/main.py`** (major rewrite)
   - Remove Ray init/shutdown
   - Add training run management commands
   - Simple process spawning

7. **`distributed/cli/config_loader.py`** (minor changes)
   - Remove Ray-specific config
   - Add MLflow config section

8. **`distributed/metrics.py`** (moderate changes)
   - Add system metrics (CPU, memory, GPU, GPU memory)
   - Add psutil/pynvml integration
   - Keep Prometheus HTTP server

9. **`configs/distributed.yaml`** (moderate changes)
   - Remove `ray:` section
   - Add `mlflow:` section
   - Add `head:` section for coordinator host

10. **`requirements.txt`** (changes)
    - Remove: `ray[default,client]`
    - Add: `mlflow`, `psutil`, `pynvml` (optional for GPU)

### New Files

1. **`distributed/coordinator/redis_state.py`**
   - Redis state management utilities
   - Atomic operations, key schema, TTL management

2. **`distributed/coordinator/run_manager.py`**
   - Training run lifecycle management
   - Start/pause/resume/stop runs
   - Checkpoint management
   - MLflow integration

3. **`tools/grafana_cluster_dashboard.json`**
   - New dashboard for cluster health
   - Node CPU/memory/GPU panels
   - Worker status panel

4. **`tools/grafana_training_dashboard.json`**
   - Training-specific dashboard
   - Loss curves, games/min, buffer size
   - Links to MLflow for detailed experiment view

### Test Updates

1. **`tests/test_coordinator.py`** (major rewrite)
   - Remove Ray fixtures
   - Test with mock Redis
   - Test MLflow integration

2. **`tests/test_workers.py`** (major rewrite)
   - Remove Ray fixtures
   - Test standalone worker lifecycle
   - Test Redis heartbeat

3. **`tests/conftest.py`** (changes)
   - Remove Ray init/shutdown
   - Add Redis mock fixtures
   - Add MLflow mock fixtures

### Script Updates

1. **`scripts/start_all_head.sh`**
   - Remove Ray head start
   - Add MLflow server start
   - Start coordinator as Python process

2. **`scripts/start_worker.sh`**
   - Remove Ray join
   - Simple Python process start
   - No cluster connection needed

3. **`scripts/stop_all.sh`**
   - Remove Ray stop
   - Kill Python processes by name

4. **`scripts/status.sh`**
   - Query Redis for worker status
   - Show MLflow run info

## Implementation Order

### Phase 1: Core Infrastructure
1. Create `redis_state.py` with key schema and utilities
2. Refactor `coordinator/head_node.py` to standalone class
3. Update `base_worker.py` to Redis-based coordination
4. Update tests for coordinator (mock Redis)

### Phase 2: Workers
5. Update `game_worker.py` to standalone
6. Update `training_worker.py` with MLflow logging
7. Update `eval_worker.py` to standalone
8. Update worker tests

### Phase 3: CLI & Scripts
9. Refactor `cli/main.py` - remove Ray, add run management
10. Update all shell scripts
11. Update `config_loader.py`

### Phase 4: Monitoring
12. Add system metrics collection (psutil/pynvml)
13. Create Grafana cluster dashboard
14. Create Grafana training dashboard
15. Update Prometheus config

### Phase 5: MLflow Integration
16. Create `run_manager.py`
17. Add checkpoint resume functionality
18. Add run cleanup commands
19. Update training worker with MLflow logging

### Phase 6: Cleanup & Docs
20. Remove all Ray imports
21. Update `requirements.txt`
22. Update `CLAUDE.md` with new architecture
23. Update config file documentation
24. Final test pass

## System Metrics to Collect

Each worker exports via Prometheus:

```
# System metrics
bgai_system_cpu_percent{worker_id="...", hostname="..."}
bgai_system_memory_percent{worker_id="...", hostname="..."}
bgai_system_memory_bytes{worker_id="...", hostname="..."}

# GPU metrics (if available)
bgai_gpu_utilization_percent{worker_id="...", gpu_id="0"}
bgai_gpu_memory_used_bytes{worker_id="...", gpu_id="0"}
bgai_gpu_memory_total_bytes{worker_id="...", gpu_id="0"}
bgai_gpu_temperature_celsius{worker_id="...", gpu_id="0"}

# Worker metrics (existing + enhanced)
bgai_games_total{worker_id="...", worker_type="game"}
bgai_games_per_minute{worker_id="..."}
bgai_model_version{worker_id="...", worker_type="game"}
bgai_worker_status{worker_id="...", status="working"}
```

## Config Changes

```yaml
# New config structure (distributed.yaml)

# Remove ray: section entirely

# Add head node config
head:
  host: "192.168.20.40"      # Head node IP
  redis_port: 6379
  mlflow_port: 5000
  coordinator_port: 8080     # REST API (optional, for status)

# Add MLflow config
mlflow:
  tracking_uri: "http://192.168.20.40:5000"
  experiment_name: "bgai-training"
  artifact_location: "./mlruns"
```

## Testing Strategy

1. **Unit tests** - Mock Redis, test individual components
2. **Integration tests** - Real Redis (localhost), test worker lifecycle
3. **Manual testing** - Multi-node with Mac worker

```bash
# Run unit tests
pytest tests/ -v -k "not integration"

# Run integration tests (requires Redis)
pytest tests/ -v -k "integration"
```

## Rollback Plan

If issues arise:
1. Keep Ray code in separate branch
2. Can revert by switching branches
3. No data migration needed (Redis buffer format unchanged)

## Success Criteria

- [ ] All tests pass
- [ ] Mac worker connects and generates games
- [ ] Training runs tracked in MLflow
- [ ] Grafana shows node metrics
- [ ] Can resume from checkpoint
- [ ] Can clean old training runs
- [ ] No Ray imports in codebase
