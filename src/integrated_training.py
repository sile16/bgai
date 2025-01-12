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
#from network-architecture import BackgammonNetwork
from mcts import AdvancedBackgammonEvaluator, BackgammonNetwork
from integrated_training import GameBuffer, SelfPlayWorker

import torch.nn.functional as F

class IntegratedTrainer:
    """Training pipeline with integrated evaluation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network and optimizer
        self.network = BackgammonNetwork().to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        # Initialize evaluator
        self.evaluator = AdvancedBackgammonEvaluator(
            self.network,
            save_dir=os.path.join(config["save_dir"], "evaluation")
        )
        
        # Initialize game buffer and workers
        self.game_buffer = GameBuffer(config["buffer_size"])
        self.workers = []
        
        # Training metrics
        self.metrics = {
            "train_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "eval_metrics": [],
            "checkpoint_metrics": []
        }
        
        # Best model tracking
        self.best_model = {
            "win_rate": 0.0,
            "epoch": 0,
            "path": None
        }
        
    def train(self):
        """Main training loop with integrated evaluation."""
        print(f"Starting training on {self.device}")
        
        # Start self-play workers
        self._start_workers()
        
        # Wait for initial games
        self._wait_for_initial_games()
        
        # Initial evaluation
        print("Performing initial evaluation...")
        initial_metrics = self._evaluate_current_model()
        self.metrics["eval_metrics"].append(initial_metrics)
        
        # Main training loop
        for epoch in range(self.config["num_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            train_metrics = self._train_epoch()
            self._update_metrics(train_metrics)
            
            # Periodic evaluation
            if (epoch + 1) % self.config["eval_interval"] == 0:
                print("\nPerforming periodic evaluation...")
                eval_metrics = self._evaluate_current_model()
                self.metrics["eval_metrics"].append(eval_metrics)
                
                # Update best model if needed
                if eval_metrics["smart_ai_winrate"] > self.best_model["win_rate"]:
                    self._save_best_model(epoch, eval_metrics)
                
                # Visualize progress
                self.evaluator.visualize_training_progress()
            
            # Save checkpoint
            if (epoch + 1) % self.config["save_interval"] == 0:
                self._save_checkpoint(epoch)
            
            # Learning rate scheduling
            self._update_learning_rate(epoch)
            
            # Early stopping check
            if self._check_early_stopping():
                print("\nEarly stopping triggered!")
                break
                
        # Final evaluation
        print("\nPerforming final evaluation...")
        final_metrics = self._evaluate_current_model()
        self.metrics["eval_metrics"].append(final_metrics)
        
        # Save final results
        self._save_final_results()
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.network.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(range(self.config["batches_per_epoch"]), 
                          desc="Training batches")
        
        for _ in progress_bar:
            # Sample batch
            batch = self.game_buffer.sample_batch(self.config["batch_size"])
            
            # Train on batch
            policy_loss, value_loss = self._train_batch(batch)
            loss = policy_loss + value_loss
            
            # Update metrics
            total_loss += loss
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'policy': f'{policy_loss:.4f}',
                'value': f'{value_loss:.4f}'
            })
            
        return {
            "train_loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches
        }
        
    def _train_batch(self, batch: List[Dict]) -> Tuple[float, float]:
        """Train on a single batch."""
        self.optimizer.zero_grad()
        
        # Prepare batch data
        states = torch.FloatTensor([pos["state"] for pos in batch]).to(self.device)
        target_policies = torch.FloatTensor([pos["policy"] for pos in batch]).to(self.device)
        target_values = torch.FloatTensor([pos["value"] for pos in batch]).to(self.device)
        
        # Forward pass
        policy_logits, values = self.network(states)
        
        # Calculate losses
        policy_loss = self._policy_loss(policy_logits, target_policies)
        value_loss = self._value_loss(values, target_values)
        
        # Combined loss and backward pass
        total_loss = policy_loss + value_loss
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 
            self.config.get("max_grad_norm", 1.0)
        )
        
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def _evaluate_current_model(self) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        # Basic metrics against different opponents
        opponent_results = {
            f"{opp}_metrics": self.evaluator.evaluate_against_opponent(opp, 
                num_games=self.config["eval_games"])
            for opp in ['smart_ai', 'random_ai']
        }
        
        # Position quality evaluation
        position_results = self.evaluator.evaluate_critical_positions(
            num_positions=self.config["eval_positions"]
        )
        
        # Endgame evaluation
        endgame_results = self.evaluator.evaluate_endgame_strength(
            num_positions=self.config["eval_positions"]
        )
        
        # Combine all metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            **opponent_results,
            "position_metrics": position_results,
            "endgame_metrics": endgame_results
        }
        
        return metrics
    
    def _save_best_model(self, epoch: int, metrics: Dict):
        """Save the best performing model."""
        self.best_model["win_rate"] = metrics["smart_ai_winrate"]
        self.best_model["epoch"] = epoch
        
        path = os.path.join(self.config["save_dir"], f"best_model_epoch_{epoch}.pt")
        self.best_model["path"] = path
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        path = os.path.join(self.config["save_dir"], f"checkpoint_epoch_{epoch}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
        }, path)
    
    def _update_learning_rate(self, epoch: int):
        """Update learning rate according to schedule."""
        if "lr_schedule" in self.config:
            for milestone, lr in self.config["lr_schedule"].items():
                if epoch == int(milestone):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print(f"Learning rate updated to {lr}")
    
    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.metrics["eval_metrics"]) < self.config["patience"]:
            return False
            
        recent_metrics = self.metrics["eval_metrics"][-self.config["patience"]:]
        win_rates = [m["smart_ai_winrate"] for m in recent_metrics]
        
        # Check if win rate hasn't improved
        best_recent = max(win_rates)
        if best_recent <= self.best_model["win_rate"]:
            return True
            
        return False
    
    def _save_final_results(self):
        """Save final training results and metrics."""
        results = {
            "config": self.config,
            "metrics": self.metrics,
            "best_model": self.best_model,
            "final_evaluation": self.metrics["eval_metrics"][-1]
        }
        
        path = os.path.join(self.config["save_dir"], "training_results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
            
    @staticmethod
    def _policy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate policy loss (cross entropy with legal move masking)."""
        return -torch.mean(torch.sum(targets * F.log_softmax(logits, dim=1), dim=1))
    
    @staticmethod
    def _value_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate value loss (MSE)."""
        return F.mse_loss(predictions.squeeze(-1), targets)
    
    def _start_workers(self):
        """Start self-play workers."""
        for _ in range(self.config["num_workers"]):
            worker = SelfPlayWorker(self.network, self.game_buffer, self.config)
            worker.start()
            self.workers.append(worker)
            
    def _wait_for_initial_games(self):
        """Wait for initial games to be generated."""
        print("Waiting for initial games...")
        while len(self.game_buffer.buffer) < self.config["min_games_to_start"]:
            games = len(self.game_buffer.buffer)
            print(f"\rGames generated: {games}/{self.config['min_games_to_start']}", 
                  end='')
            time.sleep(1)
        print("\nInitial games generated!")

# Example configuration
config = {
    # Training parameters
    "num_epochs": 1000,
    "batches_per_epoch": 100,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    
    # Self-play parameters
    "num_workers": 4,
    "buffer_size": 10000,
    "min_games_to_start": 100,
    
    # Evaluation parameters
    "eval_interval": 10,
    "eval_games": 50,
    "eval_positions": 100,
    
    # Saving parameters
    "save_dir": "training_results",
    "save_interval": 10,
    
    # Early stopping
    "patience": 20,
    
    # Learning rate schedule
    "lr_schedule": {
        "200": 0.0005,
        "400": 0.0001,
        "600": 0.00005
    }
}

# Usage example:
if __name__ == "__main__":
    trainer = IntegratedTrainer(config)
    trainer.train()