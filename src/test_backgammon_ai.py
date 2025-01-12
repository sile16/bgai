# test_backgammon_ai.py
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
import torch.nn as nn
import os
import shutil

from network_config import NetworkConfig
from network_architecture import BackgammonNetwork, NetworkOutput
from unified_training import UnifiedTrainer, TrainingConfig, GameBuffer
from metrics_visualization import MetricsTracker

class TestEvaluation:
    """Test suite for evaluation system."""
    
    @pytest.fixture
    def network(self):
        return BackgammonNetwork(NetworkConfig())
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create test directory."""
        test_dir = tmp_path / "test_outputs"
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir
        
    @pytest.fixture
    def trainer(self, network, temp_dir):
        config = TrainingConfig(
            num_epochs=2,
            batches_per_epoch=2,
            batch_size=4,
            eval_games=2,
            eval_positions=5,
            save_dir=str(temp_dir)
        )
        return UnifiedTrainer(network, config)
        
    def test_evaluate_model(self, trainer):
        """Test basic model evaluation."""
        eval_metrics = trainer._evaluate_model()
        
        # Check required metrics exist
        assert isinstance(eval_metrics, dict)
        assert "win_rate" in eval_metrics
        assert "draw_rate" in eval_metrics
        assert "position_accuracy" in eval_metrics
        
        # Check metric ranges
        assert 0 <= eval_metrics["win_rate"] <= 1
        assert 0 <= eval_metrics["draw_rate"] <= 1
        assert 0 <= eval_metrics["position_accuracy"] <= 1
        
    def test_evaluation_callback(self, trainer):
        """Test evaluation during training."""
        trainer._evaluate_and_save(epoch=0)
        
        # Check metrics were recorded
        assert len(trainer.metrics["eval_metrics"]) > 0
        last_eval = trainer.metrics["eval_metrics"][-1]
        assert "win_rate" in last_eval
        
    def test_early_stopping(self, trainer):
        """Test early stopping based on evaluation metrics."""
        # Simulate declining performance
        trainer.metrics["eval_metrics"] = [
            {"win_rate": 0.5 - i * 0.1} 
            for i in range(trainer.config.patience + 1)
        ]
        
        assert trainer._check_early_stopping()

class TestBackgammonNetwork:
    """Test suite for the neural network architecture."""
    
    @pytest.fixture
    def network_config(self):
        return NetworkConfig(
            input_channels=30,
            input_size=24,
            conv_channels=128,
            policy_channels=64,
            value_channels=32,
            hidden_size=256,
            num_blocks=3,
            kernel_size=3,
            action_space_size=7128
        )
        
    @pytest.fixture
    def network(self, network_config):
        return BackgammonNetwork(network_config)
        
    def test_network_initialization(self, network, network_config):
        """Test network initialization and architecture."""
        assert isinstance(network, BackgammonNetwork)
        assert network.input_shape == (network_config.input_channels, network_config.input_size)
        
        # Test layer initialization
        assert len(network.conv_blocks) == network_config.num_blocks
        assert isinstance(network.policy_conv, nn.Conv1d)
        assert isinstance(network.value_conv, nn.Conv1d)
        
    def test_forward_pass(self, network):
        """Test network forward pass with valid input."""
        batch_size = 32
        x = torch.randn(batch_size, 30, 24)
        
        policy_logits, value = network(x)
        
        assert policy_logits.shape == (batch_size, 7128)
        assert value.shape == (batch_size, 1)
        assert torch.all(value >= -1) and torch.all(value <= 1)
        
        # Test output ranges
        assert not torch.isnan(policy_logits).any()
        assert not torch.isnan(value).any()
        
    def test_input_validation(self, network):
        """Test input validation for invalid shapes."""
        with pytest.raises(ValueError, match="Expected 3D input tensor"):
            x = torch.randn(32, 30)
            network(x)
            
        with pytest.raises(ValueError, match="Expected input shape"):
            x = torch.randn(32, 20, 30)
            network(x)
            
    def test_initial_inference(self, network):
        """Test initial inference output structure."""
        x = torch.randn(1, 30, 24)
        output = network.initial_inference(x)
        
        assert isinstance(output, NetworkOutput)
        assert isinstance(output.value, float)
        assert output.reward == 0.0
        assert isinstance(output.policy_logits, torch.Tensor)
        assert output.policy_logits.shape == (1, 7128)
        assert output.hidden_state is None
        
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_processing(self, network, batch_size):
        """Test network processing different batch sizes."""
        x = torch.randn(batch_size, 30, 24)
        policy_logits, value = network(x)
        
        assert policy_logits.shape == (batch_size, 7128)
        assert value.shape == (batch_size, 1)
        
    def test_gradient_flow(self, network):
        """Test gradient flow through the network."""
        x = torch.randn(4, 30, 24)
        policy_logits, value = network(x)
        
        loss = policy_logits.mean() + value.mean()
        loss.backward()
        
        # Check if gradients are computed
        for param in network.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

class TestUnifiedTrainer:
    """Test suite for the training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create test directory within the test folder structure."""
        test_dir = Path(__file__).parent / "test_outputs" / "checkpoints"
        test_dir.mkdir(parents=True, exist_ok=True)
        yield test_dir
        # Clean up test files but keep the directory
        for file in test_dir.glob("*"):
            if file.is_file():
                file.unlink()
        
    @pytest.fixture
    def training_config(self):
        config = TrainingConfig(
            num_epochs=2,
            batches_per_epoch=2,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=1e-4,
            max_grad_norm=1.0,
            buffer_size=1000,
            min_games_to_start=10
        )
        return config
        
    @pytest.fixture
    def trainer(self, training_config, temp_dir):
        network = BackgammonNetwork(NetworkConfig())
        trainer = UnifiedTrainer(network, training_config)
        trainer.config.save_dir = str(temp_dir)
        return trainer
        
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert isinstance(trainer.network, BackgammonNetwork)
        assert trainer.optimizer is not None
        assert isinstance(trainer.game_buffer, GameBuffer)
        assert len(trainer.workers) == 0
        assert trainer.metrics is not None
        
    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_train_batch(self, trainer, batch_size):
        """Test training on a single batch."""
        batch = [
            {
                "state": torch.randn(30, 24),
                "policy": torch.zeros(7128).scatter_(0, torch.randint(0, 7128, (1,)), 1),  # One-hot encoded
                "value": torch.tensor(0.5)
            }
            for _ in range(batch_size)
        ]
        
        metrics = trainer._train_batch(batch)
        
        assert set(metrics.keys()) == {"total_loss", "policy_loss", "value_loss"}
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(not np.isnan(v) for v in metrics.values())
        
    def test_save_load_checkpoint(self, trainer, temp_dir):
        """Test checkpoint saving and loading."""
        # Train for a bit
        batch = [
            {
                "state": torch.randn(30, 24),
                "policy": torch.randn(7128).softmax(0),
                "value": torch.tensor(0.5)
            }
            for _ in range(4)
        ]
        trainer._train_batch(batch)
        
        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "checkpoint_0.pt"
        eval_metrics = {"win_rate": 0.5, "draw_rate": 0.1}
        trainer._save_checkpoint(0, eval_metrics)
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in trainer.network.named_parameters()}
        
        # Modify network parameters
        for param in trainer.network.parameters():
            param.data += torch.randn_like(param.data) * 0.1
            
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Verify parameters are restored
        for name, param in trainer.network.named_parameters():
            assert torch.allclose(param, initial_params[name])
            
    def test_early_stopping(self, trainer):
        """Test early stopping mechanism."""
        # Mock poor performance
        trainer.metrics["eval_metrics"] = [
            {"win_rate": 0.5 - i * 0.1} for i in range(trainer.config.patience + 1)
        ]
        
        assert trainer._check_early_stopping()
        
        # Mock improving performance
        trainer.metrics["eval_metrics"] = [
            {"win_rate": 0.5 + i * 0.1} for i in range(trainer.config.patience + 1)
        ]
        
        assert not trainer._check_early_stopping()
        
    @patch('unified_training.tqdm')
    def test_train_epoch(self, mock_tqdm, trainer):
        """Test training for one epoch."""
        mock_tqdm.return_value = range(trainer.config.batches_per_epoch)
        
        # Mock game buffer
        trainer.game_buffer = MagicMock()
        trainer.game_buffer.sample_batch.return_value = [
            {
                "state": torch.randn(30, 24),
                "policy": torch.randn(7128).softmax(0),
                "value": torch.tensor(0.5)
            }
            for _ in range(trainer.config.batch_size)
        ]
        
        metrics = trainer._train_epoch()
        
        assert set(metrics.keys()) == {"total_loss", "policy_loss", "value_loss"}
        assert trainer.game_buffer.sample_batch.call_count == trainer.config.batches_per_epoch

class TestGameBuffer:
    """Test suite for the game buffer."""
    
    @pytest.fixture
    def game_buffer(self):
        return GameBuffer(capacity=1000)
        
    def test_buffer_initialization(self, game_buffer):
        """Test buffer initialization."""
        assert game_buffer.capacity == 1000
        assert len(game_buffer.buffer) == 0
        
    def test_add_game(self, game_buffer):
        """Test adding games to buffer."""
        game = [{"state": torch.randn(30, 24)} for _ in range(10)]
        game_buffer.add_game(game)
        
        assert len(game_buffer.buffer) == 1
        assert len(game_buffer.buffer[0]) == 10
        
    def test_sample_batch(self, game_buffer):
        """Test sampling batches from buffer."""
        # Add multiple games
        for _ in range(5):
            game = [{"state": torch.randn(30, 24)} for _ in range(10)]
            game_buffer.add_game(game)
            
        batch = game_buffer.sample_batch(batch_size=16)
        assert len(batch) == 16
        assert all(isinstance(pos, dict) for pos in batch)
        
    def test_buffer_capacity(self, game_buffer):
        """Test buffer capacity limit."""
        for i in range(2000):  # More than capacity
            game = [{"state": torch.randn(30, 24)}]
            game_buffer.add_game(game)
            
        assert len(game_buffer.buffer) == 1000  # Should be limited to capacity

class TestMetricsTracker:
    """Test suite for metrics tracking and visualization."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create and clean test directory."""
        test_dir = Path(__file__).parent / "test_outputs"
        metrics_dir = test_dir / "metrics"
        
        # Clean up existing files before each test
        if metrics_dir.exists():
            for file in metrics_dir.glob("*"):
                if file.is_dir():
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    try:
                        file.unlink(missing_ok=True)
                    except PermissionError:
                        pass
            try:
                metrics_dir.rmdir()
            except (PermissionError, OSError):
                pass
                
        # Create fresh directories
        metrics_dir.mkdir(parents=True, exist_ok=True)
        yield test_dir

        # Cleanup after test
        if metrics_dir.exists():
            for file in metrics_dir.glob("*"):
                if file.is_dir():
                    shutil.rmtree(file, ignore_errors=True)
                else:
                    try:
                        file.unlink(missing_ok=True)
                    except PermissionError:
                        pass
        
    @pytest.fixture
    def metrics_tracker(self, temp_dir):
        """Initialize metrics tracker with test directory."""
        metrics_dir = temp_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tracker = MetricsTracker(save_dir=str(metrics_dir))
        tracker._is_test = True  # Mark as test instance
        yield tracker
        tracker.cleanup()  # Cleanup after each test
            
    def test_metrics_initialization(self, metrics_tracker):
        """Test metrics tracker initialization."""
        assert all(category in metrics_tracker.metrics for category in 
                  ["training", "evaluation", "game_stats", "performance"])
        
    def test_update_training_metrics(self, metrics_tracker):
        """Test updating training metrics."""
        metrics = {
            "loss": 0.5,
            "policy_loss": 0.3,
            "value_loss": 0.2
        }
        
        metrics_tracker.update_training_metrics(metrics, step=0)
        
        for key in metrics:
            assert len(metrics_tracker.metrics["training"][key]) == 1
            assert metrics_tracker.metrics["training"][key][0] == metrics[key]
            
    def test_create_training_plots(self, metrics_tracker, temp_dir):
        """Test plot creation."""
        # Add some dummy metrics
        metrics_tracker.update_training_metrics(
            {"loss": 0.5, "policy_loss": 0.3, "value_loss": 0.2}, step=0
        )
        metrics_tracker.update_eval_metrics(
            {"win_rate": 0.6, "position_accuracy": 0.7}, step=0
        )

        metrics_tracker.create_training_plots()

        # Check if plot file was created in the metrics tracker's save directory
        plot_files = list(metrics_tracker.save_dir.glob("training_summary_*.png"))
        assert len(plot_files) == 1

    def test_save_load_metrics(self, metrics_tracker, temp_dir):
        """Test saving and loading metrics."""
        # Add some metrics
        metrics_tracker.update_training_metrics(
            {"loss": 0.5, "policy_loss": 0.3}, step=0
        )

        # Save metrics
        metrics_tracker.save_metrics()

        # Wait a brief moment to ensure file is written
        import time
        time.sleep(0.1)

        # Find the saved metrics file in the metrics tracker's save directory
        metrics_files = list(metrics_tracker.save_dir.glob("metrics_*.json"))
        assert len(metrics_files) == 1

        # Create new tracker and load metrics
        new_tracker = MetricsTracker(save_dir=str(metrics_tracker.save_dir))
        new_tracker.load_metrics(metrics_files[0])

        # Compare metrics
        assert new_tracker.metrics["training"]["loss"] == metrics_tracker.metrics["training"]["loss"]
        
    def test_create_evaluation_report(self, metrics_tracker):
        """Test evaluation report generation."""
        # Add some metrics
        metrics_tracker.update_training_metrics({"loss": 0.5}, step=0)
        metrics_tracker.update_eval_metrics({"win_rate": 0.6}, step=0)
        
        report = metrics_tracker.create_evaluation_report()
        
        assert isinstance(report, str)
        assert "Training Statistics" in report
        assert "Performance Analysis" in report

if __name__ == "__main__":
    filename = "test_backgammon_ai.py"
    if not os.path.exists(filename):
        if os.path.exists("src"):
            filename = "src/" + filename
    pytest.main(["-v", filename])
