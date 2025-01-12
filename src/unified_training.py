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
            "train_metrics": {
                "total_loss": [],
                "policy_loss": [],
                "value_loss": []
            },
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
        
        for _ in tqdm(range(self.config.batches_per_epoch), desc="Training"):
            batch = self.game_buffer.sample_batch(self.config.batch_size)
            batch_metrics = self._train_batch(batch)
            
            # Update metrics
            for k, v in batch_metrics.items():
                metrics[k] += v
            metrics["num_batches"] += 1
        
        # Return average metrics
        num_batches = metrics.pop("num_batches")
        return {k: v/num_batches for k, v in metrics.items()}

    def _train_batch(self, batch: List[Dict]) -> Dict[str, float]:
        """Train on a single batch."""
        try:
            self.optimizer.zero_grad()
            
            # Prepare batch data
            states = torch.stack([
                torch.FloatTensor(pos["state"]) for pos in batch
            ]).to(self.device)  # Shape: (batch_size, 30, 24)
            
            target_policies = torch.stack([pos["policy"] for pos in batch]).to(self.device)
            target_values = torch.stack([pos["value"] for pos in batch]).squeeze().to(self.device)
            
            # Validate tensor shapes
            batch_size = len(batch)
            assert states.shape[0] == batch_size, f"Expected states batch size {batch_size}, got {states.shape[0]}"
            assert target_policies.shape[0] == batch_size, f"Expected policies batch size {batch_size}, got {target_policies.shape[0]}"
            assert target_values.shape[0] == batch_size, f"Expected values batch size {batch_size}, got {target_values.shape[0]}"
            
            # Forward pass
            policy_logits, values = self.network(states)
            
            # Calculate losses
            policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(policy_logits, dim=1), dim=1))
            value_loss = F.mse_loss(values.squeeze(), target_values)
            total_loss = policy_loss + value_loss
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # Detach and convert to float for metrics
            return {
                "total_loss": total_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "value_loss": value_loss.detach().item()
            }
            
        except RuntimeError as e:
            print(f"Training batch failed with error: {str(e)}")
            print(f"Batch shapes - States: {states.shape}, Policies: {target_policies.shape}, Values: {target_values.shape}")
            raise
    
    def _update_learning_rate(self, epoch: int) -> None:
        """Update learning rate according to schedule if defined."""
        if not hasattr(self.config, 'lr_schedule') or self.config.lr_schedule is None:
            return  # No schedule defined, keep initial learning rate
            
        # Get current schedule if it exists
        schedule = self.config.lr_schedule
        
        if isinstance(schedule, dict):
            # Dictionary schedule format: {epoch: learning_rate}
            for schedule_epoch in sorted(schedule.keys(), reverse=True):
                if epoch >= schedule_epoch:
                    new_lr = schedule[schedule_epoch]
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    break
        elif callable(schedule):
            # Function schedule format: schedule(epoch) -> learning_rate
            new_lr = schedule(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

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
            if key in self.metrics["train_metrics"]:
                self.metrics["train_metrics"][key].append(value)

    def _save_final_results(self) -> None:
        """Save final training results and generate summary."""
        print("\nSaving final training results...")
        
        # Create final results directory
        save_dir = Path(self.config.save_dir)
        final_dir = save_dir / "final_results"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Save final model
        final_model_path = final_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'final_metrics': {
                'train_metrics': self.metrics['train_metrics'],
                'eval_metrics': self.metrics['eval_metrics'][-1] if self.metrics['eval_metrics'] else {}
            }
        }, final_model_path)
        
        # Save full metrics history
        metrics_path = final_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)  # default=str handles any non-serializable objects
        
        # Generate and save training summary
        summary = self._generate_training_summary()
        summary_path = final_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        # Save final evaluator plots
        self.evaluator.plot_metrics(save=True)
        
        print(f"Final results saved to: {final_dir}")

    def _generate_training_summary(self) -> str:
        """Generate a text summary of the training run."""
        summary = ["=== Training Summary ===\n"]
        
        # Training configuration
        summary.append("Training Configuration:")
        for key, value in vars(self.config).items():
            summary.append(f"  {key}: {value}")
        
        # Final metrics
        if self.metrics["eval_metrics"]:
            final_eval = self.metrics["eval_metrics"][-1]
            summary.append("\nFinal Evaluation Metrics:")
            for key, value in final_eval.items():
                summary.append(f"  {key}: {value:.4f}")
        
        # Training statistics
        train_metrics = self.metrics["train_metrics"]
        if train_metrics["total_loss"]:
            avg_loss = sum(train_metrics["total_loss"]) / len(train_metrics["total_loss"])
            summary.append(f"\nAverage Training Loss: {avg_loss:.4f}")
        
        # Best performance
        if self.metrics["eval_metrics"]:
            best_win_rate = max(m.get("win_rate", 0) for m in self.metrics["eval_metrics"])
            summary.append(f"\nBest Win Rate: {best_win_rate:.4f}")
        
        return "\n".join(summary)

    def _cleanup(self) -> None:
        """Cleanup resources at the end of training."""
        print("\nCleaning up...")
        
        # Stop worker threads
        for worker in self.workers:
            worker.daemon = False  # Allow workers to complete current game
            if hasattr(worker, 'stop'):
                worker.stop()
        
        # Clean up evaluator
        if hasattr(self.evaluator, 'cleanup'):
            self.evaluator.cleanup()
            
        print("Cleanup complete.")