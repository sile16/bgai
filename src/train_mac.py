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
    conv_channels=64,  # Reduced from 128
    policy_channels=32,  # Reduced from 64
    value_channels=16,  # Reduced from 32
    hidden_size=128,  # Reduced from 256
    num_blocks=2,  # Reduced from 3
    kernel_size=3,
    action_space_size=7128
)

training_config = TrainingConfig(
    num_epochs=10,  # Start small
    batches_per_epoch=20,
    batch_size=32,  # Reduced from 128
    learning_rate=0.001,
    weight_decay=1e-4,
    max_grad_norm=1.0,
    num_workers=2,  # Reduced for MacBook
    buffer_size=1000,  # Reduced from 10000
    min_games_to_start=10,  # Reduced from 100
    eval_interval=2,
    eval_games=10,  # Reduced from 50
    eval_positions=20,  # Reduced from 100
    save_interval=5,
    patience=5,  # Reduced from 20
    save_dir="local_training",
    # Simple learning rate schedule for testing
    lr_schedule={
        0: 0.001,
        5: 0.0001  # Reduce LR halfway through training
    }
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