import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
from typing import List, Dict, Tuple
import threading
import queue
import time
from backgammon_env import BackgammonEnv
from mcts import BackgammonMCTS, MCTSConfig
from network_architecture import BackgammonNetwork
import torch.nn.functional as F

class GameBuffer:
    """Stores game trajectories for training."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        
    def add_game(self, game_history: List[Dict]):
        """Add a game trajectory to the buffer."""
        with self.lock:
            self.buffer.append(game_history)
            
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch of positions from games."""
        with self.lock:
            games = np.random.choice(len(self.buffer), batch_size, replace=True)
            batch = []
            for game_idx in games:
                game = self.buffer[game_idx]
                # Sample random position from game
                pos_idx = np.random.randint(len(game))
                batch.append(game[pos_idx])
            return batch

class SelfPlayWorker(threading.Thread):
    """Worker thread that generates self-play games."""
    
    def __init__(self, 
                 network: torch.nn.Module,
                 game_buffer: GameBuffer,
                 config: Dict):
        super().__init__()
        self.network = network
        self.game_buffer = game_buffer
        self.config = config
        self.daemon = True
        self.game_count = 0
        self._stop_event = threading.Event()
        
    def run(self):
        while not self.stopped():
            # Play a game
            game_history = self.play_game()
            self.game_buffer.add_game(game_history)
            self.game_count += 1
            
            if self.game_count % 10 == 0:
                print(f"Worker generated {self.game_count} games")
                
    def play_game(self) -> List[Dict]:
        """Play a single game and return its history."""
        env = BackgammonEnv()
        mcts = BackgammonMCTS(MCTSConfig())
        game_history = []
        
        done = False
        while not done:
            # Get current state and properly shape it
            state = env.get_observation()  # Shape is (30, 24)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension -> (1, 30, 24)
            
            # Run MCTS simulations
            root = mcts.run(state_tensor, self.network, env.get_state()["Roller"])
            
            # Get action probabilities from visit counts
            action_probs = mcts.get_action_probabilities(root, temperature=1.0)
            
            # Convert action_probs to tensor for storage
            action_probs_tensor = torch.zeros(env.action_space_size)
            for action, prob in action_probs.items():
                action_probs_tensor[action] = prob
            
            # Store position
            game_history.append({
                "state": state,  # Store original state
                "policy": action_probs_tensor,  # Store as tensor
                "player": env.get_state()["Roller"],
                "value": 0.0  # Will be updated when game ends
            })
            
            # Select and apply action
            action = max(action_probs.items(), key=lambda x: x[1])[0]
            _, reward, done = env.step_with_action(action)
            
            if done:
                # Add game outcome to all positions
                for pos in game_history:
                    pos["value"] = torch.FloatTensor([reward if pos["player"] == 1 else -reward])
                    
        return game_history
    
    def stop(self):
        """Signal the worker to stop."""
        self._stop_event.set()
        
    def stopped(self):
        """Check if the worker has been stopped."""
        return self._stop_event.is_set()

class Trainer:
    """Main training loop."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize network and optimizer
        self.network = BackgammonNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config["learning_rate"])
        
        # Initialize game buffer and workers
        self.game_buffer = GameBuffer(config["buffer_size"])
        self.workers = []
        for _ in range(config["num_workers"]):
            worker = SelfPlayWorker(self.network, self.game_buffer, config)
            self.workers.append(worker)
            
    def train(self):
        """Main training loop."""
        # Start workers
        for worker in self.workers:
            worker.start()
            
        # Wait for initial games
        while len(self.game_buffer.buffer) < 100:
            time.sleep(1)
            
        print("Starting training...")
        
        for epoch in range(self.config["num_epochs"]):
            total_loss = 0
            num_batches = 0
            
            # Training loop
            for _ in range(self.config["batches_per_epoch"]):
                batch = self.game_buffer.sample_batch(self.config["batch_size"])
                loss = self.train_batch(batch)
                total_loss += loss
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}: Avg loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(f"checkpoint_{epoch+1}.pt")
                
    def train_batch(self, batch: List[Dict]) -> float:
        """Train on a batch of positions."""
        self.network.train()
        self.optimizer.zero_grad()
        
        # Prepare batch data
        states = torch.FloatTensor([pos["state"] for pos in batch]).to(self.device)
        target_probs = torch.FloatTensor([pos["action_probs"] for pos in batch]).to(self.device)
        target_values = torch.FloatTensor([pos["value"] for pos in batch]).to(self.device)
        
        # Network forward pass
        policy_logits, value = self.network(states)
        
        # Calculate losses
        policy_loss = -torch.mean(torch.sum(target_probs * F.log_softmax(policy_logits, dim=1), dim=1))
        value_loss = F.mse_loss(value.squeeze(-1), target_values)
        
        # Combined loss
        loss = policy_loss + value_loss
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Configuration for local training
local_config = {
    "num_workers": 4,  # Number of self-play workers
    "buffer_size": 10000,  # Number of games to store
    "batch_size": 128,
    "num_epochs": 1000,
    "batches_per_epoch": 100,
    "learning_rate": 0.001,
    "save_interval": 10
}

# Usage example:
if __name__ == "__main__":
    trainer = Trainer(local_config)
    trainer.train()