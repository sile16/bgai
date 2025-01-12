# metrics_visualization.py
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import shutil

class MetricsTracker:
    """Comprehensive metrics tracking and visualization."""
    
    def __init__(self, save_dir: str = "training_metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer with explicit log directory
        tensorboard_dir = self.save_dir / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(tensorboard_dir))
        
        # Metrics storage
        self.metrics = {
            "training": {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "learning_rate": [],
                "gradient_norm": []
            },
            "evaluation": {
                "win_rate": [],
                "draw_rate": [],
                "average_game_length": [],
                "position_accuracy": [],
                "value_accuracy": []
            },
            "game_stats": {
                "branching_factor": [],
                "decision_complexity": [],
                "game_length": [],
                "moves_considered": []
            },
            "performance": {
                "training_speed": [],  # positions/second
                "inference_speed": [],  # positions/second
                "memory_usage": [],    # MB
                "batch_time": []       # seconds
            }
        }
        
        # Initialize plots style
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')

        sns.set_palette("husl")
        
    def update_training_metrics(self, metrics: Dict[str, float], step: int):
        """Update training metrics and log to TensorBoard."""
        for key, value in metrics.items():
            if key in self.metrics["training"]:
                self.metrics["training"][key].append(value)
                self.writer.add_scalar(f"training/{key}", value, step)
                
    def update_eval_metrics(self, metrics: Dict[str, float], step: int):
        """Update evaluation metrics and log to TensorBoard."""
        for key, value in metrics.items():
            if key in self.metrics["evaluation"]:
                self.metrics["evaluation"][key].append(value)
                # Log actual value and rolling average
                self.writer.add_scalar(f"evaluation/{key}", value, step)
                if len(self.metrics["evaluation"][key]) >= 5:
                    rolling_avg = np.mean(self.metrics["evaluation"][key][-5:])
                    self.writer.add_scalar(f"evaluation/{key}_avg5", rolling_avg, step)
                
    def log_game_stats(self, stats: Dict[str, float], step: int):
        """Log game statistics."""
        for key, value in stats.items():
            if key in self.metrics["game_stats"]:
                self.metrics["game_stats"][key].append(value)
                self.writer.add_scalar(f"game_stats/{key}", value, step)
                
    def log_performance_metrics(self, metrics: Dict[str, float], step: int):
        """Log performance metrics."""
        for key, value in metrics.items():
            if key in self.metrics["performance"]:
                self.metrics["performance"][key].append(value)
                self.writer.add_scalar(f"performance/{key}", value, step)
                
    def create_training_plots(self) -> None:
        """Create and save comprehensive training visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training Losses
        plt.subplot(3, 2, 1)
        self._plot_losses()
        
        # 2. Evaluation Metrics
        plt.subplot(3, 2, 2)
        self._plot_eval_metrics()
        
        # 3. Game Statistics
        plt.subplot(3, 2, 3)
        self._plot_game_stats()
        
        # 4. Performance Metrics
        plt.subplot(3, 2, 4)
        self._plot_performance()
        
        # 5. Learning Rate & Gradient Norm
        plt.subplot(3, 2, 5)
        self._plot_training_params()
        
        # 6. Custom Metric Correlations
        plt.subplot(3, 2, 6)
        self._plot_correlations()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"training_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_losses(self):
        """Plot training losses."""
        data = self.metrics["training"]
        plt.plot(data["loss"], label='Total Loss')
        plt.plot(data["policy_loss"], label='Policy Loss')
        plt.plot(data["value_loss"], label='Value Loss')
        plt.title('Training Losses')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
    def _plot_eval_metrics(self):
        """Plot evaluation metrics."""
        data = self.metrics["evaluation"]
        plt.plot(data["win_rate"], label='Win Rate')
        plt.plot(data["position_accuracy"], label='Position Accuracy')
        plt.plot(data["value_accuracy"], label='Value Accuracy')
        plt.title('Evaluation Metrics')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True)
        
    def _plot_game_stats(self):
        """Plot game statistics."""
        data = self.metrics["game_stats"]
        plt.plot(data["branching_factor"], label='Avg Branching Factor')
        plt.plot(data["game_length"], label='Avg Game Length')
        plt.title('Game Statistics')
        plt.xlabel('Games')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
    def _plot_performance(self):
        """Plot performance metrics."""
        data = self.metrics["performance"]
        plt.plot(data["training_speed"], label='Training Speed')
        plt.plot(data["inference_speed"], label='Inference Speed')
        plt.title('Performance Metrics')
        plt.xlabel('Step')
        plt.ylabel('Positions/second')
        plt.legend()
        plt.grid(True)
        
    def _plot_training_params(self):
        """Plot training parameters."""
        data = self.metrics["training"]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(data["learning_rate"], 'b-', label='Learning Rate')
        ax2.plot(data["gradient_norm"], 'r-', label='Gradient Norm')
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Learning Rate', color='b')
        ax2.set_ylabel('Gradient Norm', color='r')
        
        plt.title('Training Parameters')
        plt.grid(True)
        
    # Modify in metrics_visualization.py
    def _plot_correlations(self):
        """Plot correlation matrix of key metrics."""
        metrics_data = {
            'win_rate': self.metrics["evaluation"].get("win_rate", []),
            'loss': self.metrics["training"].get("loss", []),
            'position_accuracy': self.metrics["evaluation"].get("position_accuracy", []),
            'value_accuracy': self.metrics["evaluation"].get("value_accuracy", [])
        }
        
        # Ensure all arrays are the same length by taking min length
        min_length = min(len(v) for v in metrics_data.values() if len(v) > 0)
        if min_length > 0:
            metrics_df = pd.DataFrame({
                k: v[:min_length] for k, v in metrics_data.items() if len(v) > 0
            })
            correlation_matrix = metrics_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Metric Correlations')
        
    def save_metrics(self):
        """Save all metrics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.save_dir / f"metrics_{timestamp}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load_metrics(self, file_path: str):
        """Load metrics from JSON file."""
        with open(file_path, 'r') as f:
            self.metrics = json.load(f)
            
    # Modify in metrics_visualization.py
    def create_evaluation_report(self) -> str:
        """Generate a detailed evaluation report."""
        report = ["# Training Evaluation Report\n"]
        
        # Training Statistics
        report.append("## Training Statistics")
        losses = self.metrics['training'].get('loss', [])
        win_rates = self.metrics['evaluation'].get('win_rate', [])
        report.append(f"Final Loss: {losses[-1]:.4f}" if losses else "No loss data available")
        report.append(f"Best Win Rate: {max(win_rates):.2%}" if win_rates else "No win rate data available")
        
        # Performance Analysis
        report.append("\n## Performance Analysis")
        training_speeds = self.metrics['performance'].get('training_speed', [])
        memory_usage = self.metrics['performance'].get('memory_usage', [])
        report.append(f"Average Training Speed: {np.mean(training_speeds):.2f} pos/sec" if training_speeds else "No training speed data")
        report.append(f"Peak Memory Usage: {max(memory_usage):.2f} MB" if memory_usage else "No memory usage data")
        
        # Join and return
        return "\n".join(report)

    def __del__(self):
        """Cleanup TensorBoard writer."""
        self.writer.close()

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.writer.close()
        except Exception:
            pass
            
        # Clean up test directories safely
        if hasattr(self, '_is_test'):
            for file in self.save_dir.glob("*"):
                if file.is_dir():
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    try:
                        file.unlink(missing_ok=True)
                    except PermissionError:
                        pass
