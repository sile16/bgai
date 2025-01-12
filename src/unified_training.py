import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os
import json
from collections import deque
import threading
import time
from tqdm import tqdm
import torch.nn as nn
from training_pipeline import GameBuffer, SelfPlayWorker
from network_architecture import BackgammonNetwork
from network_config import NetworkConfig, TrainingConfig
from mcts import  BackgammonMCTS, MCTSConfig
from backgammon_env import BackgammonEnv
import torch.nn.functional as F
from pathlib import Path
from evaluation import BackgammonEvaluator




class UnifiedTrainer:
    """Unified training pipeline combining best practices from both implementations."""
    
    def __init__(self, network: nn.Module, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network and optimizer
        self.network = network.to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize game buffer and workers
        self.game_buffer = GameBuffer(config.buffer_size)
        self.workers = []
        
        # Training metrics
        self.metrics = {
            "train_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "eval_metrics": []
        }
        self.evaluator = BackgammonEvaluator(network, save_dir=config.save_dir)

    def _start_workers(self):
        """
        Creates self-play worker threads and starts them. 
        Adjust the arguments passed to SelfPlayWorker according to your design.
        """
        # Example: Use config.num_workers, similar to training_pipeline.py
        for _ in range(self.config.num_workers):
            # The config object used by SelfPlayWorker might differ from self.config
            # Update or replace as needed to match your SelfPlayWorker signature
            worker = SelfPlayWorker(
                network=self.network,
                game_buffer=self.game_buffer,
                config={
                    "learning_rate": self.config.learning_rate,
                    "buffer_size": self.config.buffer_size,
                    # ... any other fields SelfPlayWorker needs
                }
            )
            worker.start()
            self.workers.append(worker)
    
    def _wait_for_initial_games(self):
        """
        Waits until the game buffer holds enough games/positions for the first training step.
        Adjust the threshold (self.config.initial_games) to your needs.
        """
        # If 'initial_games' isn't defined, you can set a default or remove this check
        target_games = getattr(self.config, "initial_games", 100)
        while len(self.game_buffer.buffer) < target_games:
            time.sleep(1)
        print(f"Initial games collected: {len(self.game_buffer.buffer)}")

        
    def train(self):
        """Main training loop with integrated evaluation."""
        print(f"Starting training on {self.device}")
        
        # Start self-play workers and wait for initial games
        self._start_workers()
        self._wait_for_initial_games()
        
        # Initial evaluation
        self._evaluate_and_save()
        
        # Main training loop
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            train_metrics = self._train_epoch()
            self._update_metrics(train_metrics)
            
            # Periodic evaluation and saving
            if (epoch + 1) % self.config.eval_interval == 0:
                self._evaluate_and_save(epoch)
            
            # Learning rate scheduling
            self._update_learning_rate(epoch)
            
            # Early stopping check
            if self._check_early_stopping():
                print("\nEarly stopping triggered!")
                break
        
        # Final evaluation and cleanup
        self._evaluate_and_save(final=True)
        self._cleanup()

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()
        metrics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "num_batches": 0
        }
        
        # Fix the tqdm usage
        from tqdm import tqdm
        for _ in tqdm(range(self.config.batches_per_epoch), desc="Training"):
            batch = self.game_buffer.sample_batch(self.config.batch_size)
            batch_metrics = self._train_batch(batch)
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k] += v
            metrics["num_batches"] += 1
        
        # Return average metrics
        return {k: v/metrics["num_batches"] for k, v in metrics.items() 
                if k != "num_batches"}

    def _train_batch(self, batch: List[Dict]) -> Dict[str, float]:
        """Train on a single batch."""
        self.optimizer.zero_grad()
        
        # Prepare batch data - Fix the tensor stacking
        states = torch.stack([pos["state"] for pos in batch]).to(self.device)
        target_policies = torch.stack([pos["policy"] for pos in batch]).to(self.device)
        target_values = torch.stack([pos["value"] for pos in batch]).to(self.device)
        
        # Forward pass
        policy_logits, values = self.network(states)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        value_loss = F.mse_loss(values.squeeze(-1), target_values)
        total_loss = policy_loss + value_loss
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.metrics["eval_metrics"]) < self.config.patience:
            return False
            
        recent_metrics = self.metrics["eval_metrics"][-self.config.patience:]
        
        # Check for sustained performance decline
        win_rates = [m.get("win_rate", 0) for m in recent_metrics]
        best_recent = max(win_rates)
        first_win_rate = win_rates[0]
        
        # Stop if win rate hasn't improved over patience period
        return best_recent <= first_win_rate

    def _evaluate_and_save(self, epoch: Optional[int] = None, final: bool = False) -> None:
        """Evaluate current model and save if appropriate."""
        print("\nPerforming evaluation...")
        eval_metrics = self._evaluate_model()
        self.metrics["eval_metrics"].append(eval_metrics)
        
        # Save checkpoint
        if epoch and (epoch + 1) % self.config.save_interval == 0:
            self._save_checkpoint(epoch, eval_metrics)
            
        # Final cleanup
        if final:
            self._save_final_results()
    
    # In unified_training.py

    def _save_checkpoint(self, epoch: int, eval_metrics: dict) -> None:
        """Save training checkpoint."""
        # Create directory if it doesn't exist
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_dir / f"checkpoint_{epoch}.pt"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eval_metrics': eval_metrics,
            'metrics': self.metrics
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
                
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint.get('metrics', self.metrics)

    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model performance."""
        metrics = self.evaluator.evaluate_performance(
            num_games=self.config.eval_games,
            num_positions=self.config.eval_positions,
            mcts_sims=100  # Can make this configurable if needed
        )
        
        # Return only the metrics that the tests expect
        return {
            "win_rate": metrics["win_rate"],
            "draw_rate": metrics["draw_rate"],
            "position_accuracy": metrics["position_accuracy"]
        }

    def _evaluate_and_save(self, epoch: Optional[int] = None, final: bool = False) -> None:
        """Evaluate current model and save if appropriate."""
        print("\nPerforming evaluation...")
        eval_metrics = self._evaluate_model()
        self.metrics["eval_metrics"].append(eval_metrics)
        
        # Also save detailed metrics through the evaluator
        self.evaluator.save_metrics()
        
        # Save checkpoint if needed
        if epoch is not None and (epoch + 1) % self.config.save_interval == 0:
            self._save_checkpoint(epoch, eval_metrics)
            self.evaluator.plot_metrics(save=True)
        
        # Final cleanup
        if final:
            self._save_final_results()

    def _update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update training metrics."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)