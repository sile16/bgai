import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class MCTSConfig:
    """Configuration for MCTS parameters."""
    num_simulations: int = 800
    pb_c_base: float = 19652
    pb_c_init: float = 1.25
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    discount: float = 1.0

class MinMaxStats:
    """
    Tracking min-max statistics for value scaling.
    """
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
        
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
        
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    """Node in the MCTS tree."""
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[int, 'Node'] = {}
        self.hidden_state = None
        self.reward = 0
        
    def expanded(self) -> bool:
        return len(self.children) > 0
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """
    Monte Carlo Tree Search implementation for MuZero.
    """
    def __init__(self, config: MCTSConfig):
        self.config = config
    
    def run(self, 
            root_state: np.ndarray,
            network: Any,  # Neural network for predictions
            to_play: int) -> Node:
        """
        Run MCTS algorithm to update tree statistics.
        
        Args:
            root_state: Initial game state
            network: Neural network for predictions
            to_play: Player to move (-1 or 1)
            
        Returns:
            Root node of the search tree
        """
        root = Node(0)
        root.to_play = to_play
        # Get initial predictions from network
        network_output = network.initial_inference(root_state)
        self.expand_node(root, to_play, network_output)
        
        # Add exploration noise at the root
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac
            
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            scratch_game = root_state.copy()
            search_path = [node]
            
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                
            # Expansion and evaluation
            parent = search_path[-2]
            network_output = network.recurrent_inference(parent.hidden_state, action)
            self.expand_node(node, parent.to_play * -1, network_output)
            
            self.backpropagate(search_path, network_output.value, 
                             parent.to_play, self.config.discount)
                             
        return root
        
    def select_child(self, node: Node) -> Tuple[int, Node]:
        """Select the child with the highest UCB score."""
        max_ucb = -float('inf')
        max_action = -1
        max_child = None
        
        for action, child in node.children.items():
            ucb = self.ucb_score(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                max_action = action
                max_child = child
                
        return max_action, max_child
        
    def ucb_score(self, parent: Node, child: Node) -> float:
        """Calculate UCB score for a child."""
        pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) /
                       self.config.pb_c_base) + self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = pb_c * child.prior
        value_score = child.value()
        
        return prior_score + value_score
        
    def expand_node(self, 
                   node: Node,
                   to_play: int,
                   network_output: Any) -> None:
        """Expand a node using the network's predictions."""
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        
        policy = {a: math.exp(logit) for a, logit in enumerate(network_output.policy_logits)}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)
            
    def backpropagate(self,
                      search_path: List[Node],
                      value: float,
                      to_play: int,
                      discount: float) -> None:
        """Backpropagate value through the tree."""
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            value = node.reward + discount * value

    def get_action_probabilities(self, root: Node, temperature: float = 1.0) -> Dict[int, float]:
        """
        Returns action probabilities based on visit counts.
        """
        visits = {action: child.visit_count for action, child in root.children.items()}
        if temperature == 0:
            # Greedy selection
            max_visit = max(visits.values())
            actions = [action for action, visit in visits.items() if visit == max_visit]
            probs = {action: 1.0 / len(actions) if action in actions else 0.0 
                    for action in visits.keys()}
        else:
            # Apply temperature
            total = sum(visit ** (1.0 / temperature) for visit in visits.values())
            probs = {action: (visit ** (1.0 / temperature)) / total 
                    for action, visit in visits.items()}
        return probs

    def get_search_statistics(self, root: Node) -> Dict[str, np.ndarray]:
        """
        Returns search statistics for training.
        """
        children = root.children
        actions = list(children.keys())
        
        # Get total visits for normalization
        total_visits = sum(child.visit_count for child in children.values())
        
        # Collect statistics
        visit_counts = np.zeros(max(actions) + 1)
        values = np.zeros_like(visit_counts)
        rewards = np.zeros_like(visit_counts)
        
        for action, child in children.items():
            if total_visits > 0:
                visit_counts[action] = child.visit_count / total_visits
            values[action] = child.value()
            rewards[action] = child.reward
            
        return {
            'visit_counts': visit_counts,
            'values': values,
            'rewards': rewards
        }
    
@dataclass
class DiceRoll:
    """Represents a dice roll in backgammon."""
    moves: List[int]       # List of moves (4 for doubles, 2 for non-doubles)
    probability: float     # Probability of this roll occurring
    
class DiceRollHandler:
    """Handles the 21 unique dice roll combinations in backgammon."""
    
    def __init__(self):
        """Initialize the 21 unique roll combinations with their probabilities."""
        self.rolls: List[DiceRoll] = []
        
        # Generate all unique combinations
        for i in range(1, 7):
            for j in range(i, 7):
                moves = [i, j] if i != j else [i] * 4  # 4 moves for doubles
                prob = 1/36 if i == j else 2/36  # Doubles: 1/36, Non-doubles: 2/36
                self.rolls.append(DiceRoll(moves=moves, probability=prob))
                
        # Verify probabilities sum to 1
        total_prob = sum(roll.probability for roll in self.rolls)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities sum to {total_prob}, not 1.0"
        
    def get_roll(self) -> DiceRoll:
        """Get a random dice roll.
        
        Returns:
            DiceRoll: Contains the moves available and probability of the roll
        """
        probs = [roll.probability for roll in self.rolls]
        idx = np.random.choice(len(self.rolls), p=probs)
        return self.rolls[idx]
        
class BackgammonMCTS(MCTS):
    """MCTS implementation specialized for backgammon."""
    
    def __init__(self, config: MCTSConfig):
        super().__init__(config)
        self.dice_handler = DiceRollHandler()
        
    def run(self,
            root_state: np.ndarray,
            network: Any,
            to_play: int) -> Node:
        """
        Run MCTS with backgammon dice mechanics.
        """
        root = Node(0)
        root.to_play = to_play
        
        # Get initial predictions
        network_output = network.initial_inference(root_state)
        
        # For each simulation
        for _ in range(self.config.num_simulations):
            dice_roll = self.dice_handler.get_roll()
            moves = dice_roll.moves.copy()
            
            # Expand root with legal moves for this roll
            self.expand_node(root, to_play, network_output, moves, dice_roll.probability)
            
            node = root
            search_path = [node]
            
            # Selection - following legal moves for the roll
            while node.expanded() and moves:  # Stop when no moves left
                action, node = self.select_child(node, moves)
                search_path.append(node)
                moves.remove(action)  # Use up the selected move
                
            # Expansion and evaluation
            if len(search_path) > 1:  # Only if we made at least one move
                parent = search_path[-2]
                network_output = network.recurrent_inference(parent.hidden_state, action)
                if moves:  # If we still have moves left
                    self.expand_node(node, parent.to_play * -1, network_output, moves, dice_roll.probability)
            
            self.backpropagate(search_path, network_output.value, 
                             parent.to_play, self.config.discount)
                             
        return root
    
    def select_child(self, node: Node, available_moves: List[int]) -> Tuple[int, Node]:
        """Select child node, only considering legal moves for current roll."""
        max_ucb = -float('inf')
        max_action = -1
        max_child = None
        
        for action in available_moves:  # Only consider moves we can make
            if action not in node.children:
                continue
                
            child = node.children[action]
            ucb = self.ucb_score(node, child)
            if ucb > max_ucb:
                max_ucb = ucb
                max_action = action
                max_child = child
                
        return max_action, max_child
    
    def expand_node(self,
                   node: Node,
                   to_play: int,
                   network_output: Any,
                   available_moves: List[int],
                   roll_prob: float) -> None:
        """Expand node with only legal moves for current roll."""
        node.to_play = to_play
        node.hidden_state = network_output.hidden_state
        node.reward = network_output.reward
        
        # Get policy from network
        policy = {a: math.exp(logit) for a, logit in enumerate(network_output.policy_logits)}
        policy_sum = sum(policy.values())
        
        # Only create children for legal moves given the dice roll
        for action in available_moves:
            if action in policy:
                # Adjust prior by roll probability
                prior = (policy[action] / policy_sum) * roll_prob
                node.children[action] = Node(prior)