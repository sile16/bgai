from unified_training import UnifiedTrainer
from network_config import NetworkConfig, TrainingConfig
from network_architecture import BackgammonNetwork
from pathlib import Path

# Create output directories
base_dir = Path("training_runs")
run_name = "first_run"
run_dir = base_dir / run_name

# Create directories for different outputs
(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(run_dir / "metrics").mkdir(parents=True, exist_ok=True)

# Initialize configs
network_config = NetworkConfig(
    input_channels=30,
    input_size=24,
    conv_channels=128,
    policy_channels=64,
    value_channels=32,
    hidden_size=256,
    num_blocks=3
)

training_config = TrainingConfig(
    num_epochs=1000,
    batches_per_epoch=50,
    batch_size=64,  # Smaller for MacBook
    learning_rate=0.001,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    num_workers=4,  # Adjust based on CPU cores
    buffer_size=5000,
    min_games_to_start=50,
    eval_interval=5,
    eval_games=20,
    eval_positions=50,
    save_interval=10,
    patience=20,
    save_dir=str(run_dir / "checkpoints")
)

if __name__ == "__main__":
    print(f"Starting training run: {run_name}")
    print(f"Outputs will be saved to: {run_dir}")
    
    network = BackgammonNetwork(network_config)
    trainer = UnifiedTrainer(network, training_config)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        trainer._save_checkpoint("interrupt", {})
        print("Checkpoint saved. You can resume training later.")