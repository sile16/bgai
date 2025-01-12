
from integrated_training import IntegratedTrainer

local_config = {
    "num_workers": 4,  # Adjust based on CPU cores
    "batch_size": 64,  # Smaller for MacBook
    "buffer_size": 5000,
    "eval_games": 20,
    "eval_positions": 50
}
trainer = IntegratedTrainer(local_config)
trainer.train()