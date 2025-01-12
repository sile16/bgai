# evaluation.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from backgammon_env import BackgammonEnv
from mcts import BackgammonMCTS, MCTSConfig

class BackgammonEvaluator:
    """Unified evaluator for backgammon AI performance."""
    
    def __init__(self, network: torch.nn.Module, save_dir: str = "evaluation_results"):
        self.network = network
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_history = defaultdict(list)
        
    def evaluate_performance(self, 
                           num_games: int = 100,
                           num_positions: int = 100,
                           mcts_sims: int = 100) -> Dict[str, float]:
        """Comprehensive evaluation of model performance.
        
        Args:
            num_games: Number of games to play against smart AI
            num_positions: Number of positions to analyze
            mcts_sims: Number of MCTS simulations per move
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.network.eval()
        
        metrics = {}
        
        try:
            # 1. Evaluate against smart AI
            metrics.update(self._evaluate_against_ai(num_games, mcts_sims))
            
            # 2. Evaluate position accuracy
            metrics.update(self._evaluate_positions(num_positions, mcts_sims))
            
            # 3. Evaluate endgame performance
            metrics.update(self._evaluate_endgame(num_positions // 2, mcts_sims))
            
            # Store metrics history
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for key, value in metrics.items():
                self.metrics_history[key].append((timestamp, value))
                
        finally:
            self.network.train()
            
        return metrics
    
    def _evaluate_against_ai(self, num_games: int, mcts_sims: int) -> Dict[str, float]:
        """Evaluate network against smart AI opponent."""
        env = BackgammonEnv()
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=mcts_sims))
        
        wins = 0
        draws = 0
        points_won = 0
        
        # Play games
        with torch.no_grad():
            for _ in range(num_games):
                env.reset()
                env.set_red_ai(env.SMART_AI)
                done = False
                
                while not done:
                    state = env.get_state()
                    if state["Roller"] == 1:  # White # Our turn
                        obs = env.get_observation()
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        try:
                            root = mcts.run(obs_tensor, self.network, state["Roller"])
                            action = max(mcts.get_action_probabilities(root, temperature=0.0).items(),
                                    key=lambda x: x[1])[0]
                        except Exception as e:
                            print(f"Error during MCTS: {e}")
                            action = 0  # Default action
                        _, victor, points = env.step(action)
                    else:  # AI's turn
                        _, victor, points = env.step_ai()
                        
                    done = victor is not None
                    
                if victor == 0:
                    draws += 1
                elif victor == 1:
                    wins += 1
                    points_won += points
        
        return {
            "win_rate": wins / num_games,
            "draw_rate": draws / num_games,
            "avg_points_when_winning": points_won / wins if wins > 0 else 0
        }
    
    def _evaluate_positions(self, num_positions: int, mcts_sims: int) -> Dict[str, float]:
        """Evaluate position accuracy against smart AI."""
        env = BackgammonEnv()
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=mcts_sims))
        
        correct_positions = 0
        value_correlation = []
        
        with torch.no_grad():
            for _ in range(num_positions):
                env.reset()
                state = env.get_state()
                
                # Get our network's move
                obs = env.get_observation()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                root = mcts.run(obs_tensor, self.network, state["Roller"])
                our_move = max(mcts.get_action_probabilities(root, temperature=0.0).items(),
                             key=lambda x: x[1])[0]
                our_value = root.value()
                
                # Get smart AI's move
                env.set_white_ai(env.SMART_AI)
                smart_moves = env.get_legal_moves()
                
                # Compare moves
                if our_move < len(smart_moves):
                    correct_positions += 1
                
                # Store value correlation
                if hasattr(env, 'get_position_value'):  # If available
                    true_value = env.get_position_value()
                    value_correlation.append((our_value, true_value))
        
        metrics = {
            "position_accuracy": correct_positions / num_positions
        }
        
        if value_correlation:
            correlation = np.corrcoef(np.array(value_correlation).T)[0,1]
            metrics["value_correlation"] = correlation
            
        return metrics
    
    def _evaluate_endgame(self, num_positions: int, mcts_sims: int) -> Dict[str, float]:
        """Evaluate endgame performance."""
        env = BackgammonEnv()
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=mcts_sims))
        
        wins = 0
        moves_to_win = []
        
        with torch.no_grad():
            for _ in range(num_positions):
                env.reset()
                # Set up winning endgame position
                if hasattr(env, 'setup_winning_endgame'):
                    env.setup_winning_endgame()
                    
                done = False
                moves = 0
                
                while not done and moves < 20:  # Limit moves to prevent infinite games
                    state = env.get_state()
                    obs = env.get_observation()
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    root = mcts.run(obs_tensor, self.network, state["Roller"])
                    action = max(mcts.get_action_probabilities(root, temperature=0.0).items(),
                              key=lambda x: x[1])[0]
                    _, victor, _ = env.step(action)
                    
                    done = victor is not None
                    moves += 1
                    
                if victor == 1:
                    wins += 1
                    moves_to_win.append(moves)
        
        return {
            "endgame_win_rate": wins / num_positions,
            "avg_moves_to_win": np.mean(moves_to_win) if moves_to_win else float('inf')
        }
    
    def save_metrics(self):
        """Save evaluation metrics history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f'metrics_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics_history), f, indent=2)
    
    def plot_metrics(self, save: bool = True):
        """Plot evaluation metrics over time."""
        if not self.metrics_history:
            print("No metrics to plot - metrics_history is empty")
            return
            
        print(f"Plotting {len(self.metrics_history)} metrics: {list(self.metrics_history.keys())}")
        
        # Calculate number of rows and columns needed
        n_metrics = len(self.metrics_history)
        n_cols = min(3, n_metrics)  # Max 3 columns
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        # Plot each metric
        for i, (metric, history) in enumerate(self.metrics_history.items()):
            plt.subplot(n_rows, n_cols, i+1)
            if history:  # Check if history exists
                timestamps, values = zip(*history)
                plt.plot(range(len(values)), values, marker='o')
                plt.title(metric.replace('_', ' ').title())
                plt.xlabel('Evaluation Number')
                plt.ylabel('Value')
                plt.grid(True)
            
        plt.tight_layout()
        
        try:
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.save_dir, f'metrics_plot_{timestamp}.png')
                plt.savefig(save_path)
                print(f"Saved metrics plot to: {save_path}")
        except Exception as e:
            print(f"Error saving metrics plot: {str(e)}")
        finally:
            plt.close()