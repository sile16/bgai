import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from training_pipeline import GameBuffer, SelfPlayWorker
from network_architecture import BackgammonNetwork
from network_config import NetworkConfig, TrainingConfig
from mcts import BackgammonNetwork, BackgammonMCTS, MCTSConfig
from backgammon_env import BackgammonEnv

class BackgammonEvaluator:
    """Evaluates backgammon AI performance using multiple metrics."""
    
    def __init__(self, network: torch.nn.Module, save_dir: str = "evaluation_results"):
        self.network = network
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(list)
        self.current_tournament = None
        
    def evaluate_against_opponent(self, 
                                opponent_type: str,
                                num_games: int = 100,
                                temperature: float = 0.0) -> Dict[str, float]:
        """Evaluate network against a specific opponent type.
        
        Args:
            opponent_type: One of 'smart_ai', 'random_ai', 'worst_ai'
            num_games: Number of games to play
            temperature: Temperature for move selection (0 = deterministic)
        """
        env = BackgammonEnv()
        
        # Set up opponent
        opponent_map = {
            'smart_ai': env.SMART_AI,
            'random_ai': env.RANDOM_AI,
            'worst_ai': env.WORST_AI
        }
        
        results = {
            'wins': 0,
            'losses': 0,
            'points_won': 0,
            'points_lost': 0,
            'avg_moves_per_game': 0
        }
        
        total_moves = 0
        
        for game in range(num_games):
            env.reset()
            
            # Randomly assign colors
            network_plays_white = np.random.random() < 0.5
            if network_plays_white:
                env.set_red_ai(opponent_map[opponent_type])
            else:
                env.set_white_ai(opponent_map[opponent_type])
            
            done = False
            moves_this_game = 0
            
            while not done:
                state = env.get_state()
                network_turn = (state["player"] == 1 and network_plays_white) or \
                             (state["player"] == -1 and not network_plays_white)
                
                if network_turn:
                    # Use network to select move
                    action = self._select_move(env, temperature)
                    _, victor, points = env.step(action)
                else:
                    # Let opponent make move
                    _, victor, points = env.step_ai()
                
                moves_this_game += 1
                done = victor is not None
                
                if done:
                    network_won = (victor == 1 and network_plays_white) or \
                                (victor == -1 and not network_plays_white)
                    if network_won:
                        results['wins'] += 1
                        results['points_won'] += points
                    else:
                        results['losses'] += 1
                        results['points_lost'] += points
            
            total_moves += moves_this_game
            
        # Calculate aggregate statistics
        total_games = results['wins'] + results['losses']
        results['win_rate'] = results['wins'] / total_games
        results['avg_moves_per_game'] = total_moves / total_games
        results['avg_points_when_winning'] = results['points_won'] / results['wins'] if results['wins'] > 0 else 0
        results['avg_points_when_losing'] = results['points_lost'] / results['losses'] if results['losses'] > 0 else 0
        
        # Store metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history[f'{opponent_type}_winrate'].append((timestamp, results['win_rate']))
        self.metrics_history[f'{opponent_type}_avg_points'].append(
            (timestamp, results['avg_points_when_winning'])
        )
        
        return results
    
    def evaluate_move_quality(self, 
                            num_positions: int = 100,
                            comparison_depth: int = 1000) -> Dict[str, float]:
        """Evaluate quality of move selection against deep search baseline.
        
        Args:
            num_positions: Number of positions to analyze
            comparison_depth: Depth of search for baseline comparison
        """
        env = BackgammonEnv()
        results = {
            'move_match_rate': 0,
            'value_correlation': [],
            'policy_kl_div': []
        }
        
        matching_moves = 0
        
        for _ in range(num_positions):
            env.reset()
            
            # Play random moves to reach middle game positions
            for _ in range(np.random.randint(5, 15)):
                legal_moves = env.get_legal_moves()
                if not legal_moves:
                    break
                action = np.random.choice(len(legal_moves))
                env.step(action)
                
            # Get network's move selection
            network_action, network_value = self._get_network_prediction(env)
            
            # Get deep search baseline
            baseline_action, baseline_value = self._get_baseline_prediction(env, comparison_depth)
            
            # Compare moves
            if network_action == baseline_action:
                matching_moves += 1
                
            # Store value correlation
            results['value_correlation'].append((network_value, baseline_value))
            
            # Calculate policy KL divergence
            # (simplified - just using top moves)
            results['policy_kl_div'].append(
                self._calculate_policy_kl(env, comparison_depth)
            )
            
        results['move_match_rate'] = matching_moves / num_positions
        results['value_correlation'] = np.corrcoef(
            np.array(results['value_correlation']).T
        )[0,1]
        results['policy_kl_div'] = np.mean(results['policy_kl_div'])
        
        # Store metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_history['move_match_rate'].append(
            (timestamp, results['move_match_rate'])
        )
        self.metrics_history['value_correlation'].append(
            (timestamp, results['value_correlation'])
        )
        
        return results
    
    def run_tournament(self, 
                      checkpoint_paths: List[str], 
                      games_per_pair: int = 50) -> Dict[str, float]:
        """Run tournament between different versions of the network.
        
        Args:
            checkpoint_paths: List of paths to network checkpoints
            games_per_pair: Number of games to play between each pair
        """
        results = defaultdict(lambda: defaultdict(int))
        
        for i, path1 in enumerate(checkpoint_paths):
            network1 = self._load_network(path1)
            
            for path2 in checkpoint_paths[i+1:]:
                network2 = self._load_network(path2)
                
                # Play games between the networks
                outcome = self._play_network_vs_network(
                    network1, network2, games_per_pair
                )
                
                # Store results
                results[path1][path2] = outcome
                results[path2][path1] = -outcome
                
        self.current_tournament = results
        return dict(results)
    
    def plot_training_progress(self, metric: str, save: bool = True):
        """Plot progress of a specific metric over time.
        
        Args:
            metric: Metric name to plot
            save: Whether to save the plot to disk
        """
        if metric not in self.metrics_history:
            raise ValueError(f"No data for metric: {metric}")
            
        timestamps, values = zip(*self.metrics_history[metric])
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(values)), values, marker='o')
        plt.title(f'{metric} Over Time')
        plt.xlabel('Evaluation Number')
        plt.ylabel(metric)
        plt.grid(True)
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'{metric}_progress.png'))
        plt.close()
    
    def save_metrics(self):
        """Save all metrics to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics history
        metrics_path = os.path.join(self.save_dir, f'metrics_history_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json.dump(dict(self.metrics_history), f, indent=2)
            
        # Save current tournament results if available
        if self.current_tournament:
            tournament_path = os.path.join(self.save_dir, f'tournament_{timestamp}.json')
            with open(tournament_path, 'w') as f:
                json.dump(dict(self.current_tournament), f, indent=2)
    
    def _select_move(self, env: BackgammonEnv, temperature: float) -> int:
        """Select a move using the network with optional temperature."""
        mcts = BackgammonMCTS(MCTSConfig())
        root = mcts.run(env.get_observation(), self.network, env.get_state()["player"])
        
        action_probs = mcts.get_action_probabilities(root, temperature)
        if temperature == 0:
            action = max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            actions, probs = zip(*action_probs.items())
            action = np.random.choice(actions, p=probs)
            
        return action
    
    def _get_network_prediction(self, env: BackgammonEnv) -> Tuple[int, float]:
        """Get network's move prediction and value estimate."""
        state = torch.FloatTensor(env.get_observation()).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.network(state)
            
        # Get action with highest probability
        action = torch.argmax(policy_logits).item()
        value = value.item()
        
        return action, value
    
    def _get_baseline_prediction(self, 
                               env: BackgammonEnv, 
                               search_depth: int) -> Tuple[int, float]:
        """Get deep search baseline prediction."""
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=search_depth))
        root = mcts.run(env.get_observation(), self.network, env.get_state()["player"])
        
        action_probs = mcts.get_action_probabilities(root, temperature=0)
        action = max(action_probs.items(), key=lambda x: x[1])[0]
        value = root.value()
        
        return action, value
    
    def _calculate_policy_kl(self, 
                           env: BackgammonEnv, 
                           search_depth: int) -> float:
        """Calculate KL divergence between network and baseline policies."""
        # Get network policy
        state = torch.FloatTensor(env.get_observation()).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.network(state)
        network_probs = torch.softmax(policy_logits, dim=1).squeeze()
        
        # Get baseline policy
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=search_depth))
        root = mcts.run(env.get_observation(), self.network, env.get_state()["player"])
        action_probs = mcts.get_action_probabilities(root, temperature=1.0)
        
        # Calculate KL divergence for legal moves only
        kl_div = 0
        legal_moves = env.get_legal_moves()
        for move in legal_moves:
            p = network_probs[move].item()
            q = action_probs.get(move, 1e-8)  # Small constant to avoid log(0)
            if p > 0:
                kl_div += p * np.log(p / q)
                
        return kl_div
    
    @staticmethod
    def _load_network(checkpoint_path: str) -> torch.nn.Module:
        """Load network from checkpoint."""
        network = BackgammonNetwork()
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['network_state_dict'])
        return network
        
    def _play_network_vs_network(self,
                                network1: torch.nn.Module,
                                network2: torch.nn.Module,
                                num_games: int) -> float:
        """Play games between two networks and return relative performance."""
        total_score = 0
        
        for game in range(num_games):
            score = self._play_single_game(network1, network2)
            total_score += score
            
        return total_score / num_games
    
    def _play_single_game(self,
                         network1: torch.nn.Module,
                         network2: torch.nn.Module) -> float:
        """Play a single game between two networks."""
        env = BackgammonEnv()
        done = False
        
        while not done:
            state = env.get_state()
            current_network = network1 if state["player"] == 1 else network2
            
            # Select and apply move
            action = self._select_move_for_network(current_network, env)
            _, victor, points = env.step(action)
            
            done = victor is not None
            
        # Return score from network1's perspective
        if victor == 1:
            return points
        else:
            return -points

    def _select_move_for_network(self,
                                network: torch.nn.Module,
                                env: BackgammonEnv) -> int:
        """Select a move using the given network."""
        mcts = BackgammonMCTS(MCTSConfig())
        root = mcts.run(env.get_observation(), network, env.get_state()["player"])
        action_probs = mcts.get_action_probabilities(root, temperature=0)
        return max(action_probs.items(), key=lambda x: x[1])[0]

# Example usage:
if __name__ == "__main__":
    network = BackgammonNetwork()
    evaluator = BackgammonEvaluator(network)
    
    # Evaluate against different opponents
    results_smart = evaluator.evaluate_against_opponent('smart_ai', num_games=50)
    results_random = evaluator.evaluate_against_opponent('random_ai', num_games=50)
    
    # Evaluate move quality
    move_quality = evaluator.evaluate_move_quality(num_positions=50)
    
    # Plot progress
    evaluator.plot_training_progress('smart_ai_winrate')
    evaluator.plot_training_progress('move_match_rate')
    
    # Save results
    evaluator.save_metrics()
    
    print("Smart AI Results:", results_smart)
    print("Random AI Results:", results_random)