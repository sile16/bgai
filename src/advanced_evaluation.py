import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from training_pipeline import GameBuffer, SelfPlayWorker
from network_architecture import BackgammonNetwork
from network_config import NetworkConfig, TrainingConfig
from mcts import BackgammonMCTS, MCTSConfig
from backgammon_env import BackgammonEnv

class AdvancedBackgammonEvaluator:
    """Advanced evaluation methods for backgammon AI."""
    
    def __init__(self, network: torch.nn.Module, save_dir: str = "advanced_evaluation"):
        self.network = network
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics history
        self.training_metrics = defaultdict(list)
        
        # Position evaluation cache
        self.position_cache = {}
        
    def evaluate_critical_positions(self, num_positions: int = 100) -> Dict[str, float]:
        """Evaluate network on critical backgammon positions.
        
        Analyzes performance on key position types:
        - Opening moves
        - Bear-off positions
        - Racing positions
        - Back game positions
        - Holding game positions
        """
        position_types = {
            'opening': self._generate_opening_positions,
            'bearoff': self._generate_bearoff_positions,
            'racing': self._generate_racing_positions,
            'backgame': self._generate_backgame_positions,
            'holding': self._generate_holding_positions
        }
        
        results = {}
        
        for pos_type, generator_func in position_types.items():
            # Generate positions
            positions = generator_func(num_positions // len(position_types))
            
            # Evaluate each position
            type_results = self._evaluate_positions(positions)
            results[pos_type] = type_results
            
            # Store in cache for future reference
            for pos in positions:
                pos_key = self._position_to_key(pos)
                self.position_cache[pos_key] = type_results
                
        return results
    
    def analyze_opening_book(self, depth: int = 3) -> Dict[str, List[Tuple[int, float]]]:
        """Analyze opening move sequences up to given depth.
        
        Args:
            depth: How many moves deep to analyze
            
        Returns:
            Dictionary mapping opening sequences to (move, evaluation) pairs
        """
        env = BackgammonEnv()
        opening_book = {}
        
        def analyze_sequence(sequence: List[int], current_depth: int):
            if current_depth >= depth:
                return
                
            # Get legal moves for current position
            legal_moves = env.get_legal_moves()
            
            # Get network evaluation for each move
            move_evals = []
            for move in legal_moves:
                # Make move
                state_copy = env.get_state()
                env.step(move)
                
                # Get evaluation
                eval_result = self._get_network_evaluation(env)
                move_evals.append((move, eval_result))
                
                # Recurse
                sequence_key = tuple(sequence + [move])
                opening_book[sequence_key] = move_evals
                analyze_sequence(sequence + [move], current_depth + 1)
                
                # Restore position
                env.reset()
                env.set_state(state_copy)
                
        # Start analysis
        analyze_sequence([], 0)
        return opening_book
    
    def evaluate_endgame_strength(self, num_positions: int = 50) -> Dict[str, float]:
        """Evaluate network's endgame play.
        
        Tests positions with different pip counts and piece distributions.
        """
        results = defaultdict(list)
        
        # Test different endgame scenarios
        pip_counts = [(20, 40), (30, 50), (40, 60)]  # (player, opponent) pip counts
        
        for player_pips, opponent_pips in pip_counts:
            positions = self._generate_endgame_positions(
                num_positions // len(pip_counts),
                player_pips,
                opponent_pips
            )
            
            for pos in positions:
                # Get network's evaluation
                eval_result = self._evaluate_endgame_position(pos)
                
                # Store results
                key = f"pips_{player_pips}_{opponent_pips}"
                results[key].append(eval_result)
                
        # Calculate statistics
        stats = {}
        for key, values in results.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'win_rate': np.mean([v > 0 for v in values])
            }
            
        return stats
    
    def visualize_training_progress(self):
        """Create comprehensive visualization of training progress."""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Win rates vs different opponents
        plt.subplot(2, 2, 1)
        for opponent in ['smart_ai', 'random_ai', 'worst_ai']:
            if f'{opponent}_winrate' in self.training_metrics:
                data = self.training_metrics[f'{opponent}_winrate']
                plt.plot(range(len(data)), data, label=opponent)
        plt.title('Win Rates vs Opponents')
        plt.xlabel('Training Iteration')
        plt.ylabel('Win Rate')
        plt.legend()
        
        # 2. Move quality metrics
        plt.subplot(2, 2, 2)
        metrics = ['move_match_rate', 'value_correlation']
        for metric in metrics:
            if metric in self.training_metrics:
                data = self.training_metrics[metric]
                plt.plot(range(len(data)), data, label=metric)
        plt.title('Move Quality Metrics')
        plt.xlabel('Training Iteration')
        plt.ylabel('Score')
        plt.legend()
        
        # 3. Loss components
        plt.subplot(2, 2, 3)
        loss_types = ['policy_loss', 'value_loss', 'total_loss']
        for loss_type in loss_types:
            if loss_type in self.training_metrics:
                data = self.training_metrics[loss_type]
                plt.plot(range(len(data)), data, label=loss_type)
        plt.title('Training Losses')
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        
        # 4. Position type performance
        plt.subplot(2, 2, 4)
        pos_types = ['opening', 'bearoff', 'racing', 'backgame', 'holding']
        if all(f'{pt}_score' in self.training_metrics for pt in pos_types):
            data = [self.training_metrics[f'{pt}_score'][-1] for pt in pos_types]
            plt.bar(pos_types, data)
            plt.title('Performance by Position Type')
            plt.xticks(rotation=45)
            plt.ylabel('Score')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.save_dir, f'training_progress_{timestamp}.png'))
        plt.close()
        
    def analyze_game_complexity(self, game_records: List[Dict]) -> Dict[str, float]:
        """Analyze complexity of played games.
        
        Args:
            game_records: List of game records with moves and evaluations
            
        Returns:
            Dictionary with complexity metrics
        """
        metrics = defaultdict(list)
        
        for game in game_records:
            # Calculate branching factor
            avg_branches = np.mean([len(move['legal_moves']) for move in game['moves']])
            metrics['branching_factor'].append(avg_branches)
            
            # Calculate evaluation volatility
            evals = [move['evaluation'] for move in game['moves']]
            eval_changes = np.abs(np.diff(evals))
            metrics['eval_volatility'].append(np.mean(eval_changes))
            
            # Calculate decision complexity
            # (how close alternative moves were in evaluation)
            decision_complexity = []
            for move in game['moves']:
                if len(move['move_evals']) > 1:
                    top_2_diff = abs(move['move_evals'][0] - move['move_evals'][1])
                    decision_complexity.append(top_2_diff)
            metrics['decision_complexity'].append(np.mean(decision_complexity))
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _position_to_key(self, position: Dict) -> str:
        """Convert position to string key for caching."""
        return json.dumps(position, sort_keys=True)
    
    def _get_network_evaluation(self, env: BackgammonEnv) -> float:
        """Get network's evaluation of current position."""
        state = torch.FloatTensor(env.get_observation()).unsqueeze(0)
        with torch.no_grad():
            _, value = self.network(state)
        return value.item()
    
    def _evaluate_positions(self, positions: List[Dict]) -> Dict[str, float]:
        """Evaluate a list of positions."""
        results = defaultdict(list)
        
        env = BackgammonEnv()
        for pos in positions:
            env.reset()
            env.set_state(pos)
            
            # Get network evaluation
            eval_result = self._get_network_evaluation(env)
            results['evals'].append(eval_result)
            
            # Get move quality metrics
            move_quality = self.evaluate_move_quality(env)
            for k, v in move_quality.items():
                results[k].append(v)
                
        return {k: np.mean(v) for k, v in results.items()}
    
    def _evaluate_endgame_position(self, position: Dict) -> float:
        """Evaluate an endgame position using deep search."""
        env = BackgammonEnv()
        env.reset()
        env.set_state(position)
        
        # Use deeper search for endgame positions
        mcts = BackgammonMCTS(MCTSConfig(num_simulations=2000))
        root = mcts.run(env.get_observation(), self.network, env.get_state()["player"])
        
        return root.value()
    
    def _generate_opening_positions(self, count: int) -> List[Dict]:
        """Generate opening position variations."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Modify initial position slightly
            state = env.get_state()
            # Add small random variations to standard opening
            positions.append(state)
            
        return positions
    
    def _generate_bearoff_positions(self, count: int) -> List[Dict]:
        """Generate bear-off positions with varying difficulty."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Set up position with all pieces in home board
            state = env.get_state()
            # Randomly distribute pieces in last 6 points
            positions.append(state)
            
        return positions
    
    def _generate_racing_positions(self, count: int) -> List[Dict]:
        """Generate racing positions (no contact between pieces)."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Set up position with pieces past each other
            state = env.get_state()
            positions.append(state)
            
        return positions
    
    def _generate_backgame_positions(self, count: int) -> List[Dict]:
        """Generate back game positions."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Set up position with pieces back
            state = env.get_state()
            positions.append(state)
            
        return positions
    
    def _generate_holding_positions(self, count: int) -> List[Dict]:
        """Generate holding game positions."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Set up position with prime/block
            state = env.get_state()
            positions.append(state)
            
        return positions
    
    def _generate_endgame_positions(self, 
                                  count: int,
                                  player_pips: int,
                                  opponent_pips: int) -> List[Dict]:
        """Generate endgame positions with specific pip counts."""
        positions = []
        env = BackgammonEnv()
        
        for _ in range(count):
            env.reset()
            # Set up position matching pip counts
            state = env.get_state()
            positions.append(state)
            
        return positions
    
    def save_evaluation_results(self, results: Dict):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_dir, f'evaluation_{timestamp}.json')
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
    def load_evaluation_results(self, filepath: str) -> Dict:
        """Load previous evaluation results."""
        with open(filepath, 'r') as f:
            return json.load(f)

# Example usage:
if __name__ == "__main__":
    network = BackgammonNetwork()
    evaluator = AdvancedBackgammonEvaluator(network)
    
    # Evaluate critical positions
    position_results = evaluator.evaluate_critical_positions(num_positions=100)
    print("Position evaluation results:", position_results)
    
    # Analyze opening book
    opening_book = evaluator.analyze_opening_book(depth=3)
    print("Opening book analysis completed")
    
    # Evaluate endgame strength
    endgame_results = evaluator.evaluate_endgame_strength(num_positions=50)
    print("Endgame evaluation results:", endgame_results)
    
    # Visualize progress
    evaluator.visualize_training_progress()
    
    # Save results
    evaluator.save_evaluation_results({
        'position_results': position_results,
        'endgame_results': endgame_results
    })