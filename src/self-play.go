import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GameExample:
    state: np.ndarray
    policy: np.ndarray
    value: float
    player: int

class SelfPlay:
    def __init__(self, game, mcts_config, network, num_games=1000):
        self.game = game
        self.mcts_config = mcts_config
        self.network = network
        self.num_games = num_games
        
    def generate_game(self) -> List[GameExample]:
        """Generate a single game of self-play."""
        examples = []
        game = self.game.new_game()
        mcts = MCTS(game, self.network, self.mcts_config)
        
        while not game.is_terminal():
            state = game.get_state()
            
            # Run MCTS simulations
            policy = mcts.search(self.mcts_config.num_simulations)
            
            # Store the example
            examples.append(GameExample(
                state=state,
                policy=policy,
                value=0.0,  # Will be updated when game ends
                player=game.current_player()
            ))
            
            # Select and apply move
            move_idx = np.random.choice(len(policy), p=policy)
            game.apply_move(move_idx)
            mcts.update_with_move(move_idx)
        
        # Update values based on game outcome
        value = game.get_result()
        for example in examples:
            example.value = value if example.player == 1 else -value
            
        return examples
    
    def execute_episode(self):
        """Generate games in parallel."""
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.generate_game) 
                      for _ in range(self.num_games)]
            all_examples = []
            for future in futures:
                all_examples.extend(future.result())
        
        return all_examples

class Trainer:
    def __init__(self, network, optimizer, config):
        self.network = network
        self.optimizer = optimizer
        self.config = config
        
    def train_epoch(self, examples):
        """Train on a batch of examples from self-play."""
        self.network.train()
        
        for batch in self.create_batches(examples):
            states, policies, values = batch
            
            # Forward pass
            policy_pred, value_pred = self.network(states)
            
            # Calculate loss
            policy_loss = self.policy_loss(policy_pred, policies)
            value_loss = self.value_loss(value_pred, values)
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
            
    @staticmethod
    def policy_loss(pred, target):
        return -torch.sum(target * torch.log(pred + 1e-8)) / pred.size(0)
        
    @staticmethod
    def value_loss(pred, target):
        return torch.mean((pred - target) ** 2)
