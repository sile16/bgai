import unittest
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from mcts import MCTS, MCTSConfig, Node, MinMaxStats, DiceRollHandler, BackgammonMCTS
from collections import defaultdict
import math

@dataclass
class MockNetworkOutput:
    hidden_state: np.ndarray
    reward: float
    policy_logits: List[float]
    value: float

class MockNetwork:
    def __init__(self, policy_logits: List[float], value: float):
        self.policy_logits = policy_logits
        self.value = value
        
    def initial_inference(self, state):
        return MockNetworkOutput(
            hidden_state=np.zeros(8),  # Mock hidden state
            reward=0.0,
            policy_logits=self.policy_logits,
            value=self.value
        )
        
    def recurrent_inference(self, hidden_state, action):
        return MockNetworkOutput(
            hidden_state=np.zeros(8),  # Mock hidden state
            reward=0.0,
            policy_logits=self.policy_logits,
            value=self.value
        )

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.config = MCTSConfig(
            num_simulations=100,
            pb_c_base=19652,
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,
            root_exploration_fraction=0.25,
            discount=1.0
        )
        self.mcts = MCTS(self.config)

    def test_node_initialization(self):
        """Test basic node creation and properties."""
        node = Node(prior=0.5)
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.prior, 0.5)
        self.assertEqual(node.value_sum, 0)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.expanded())

    def test_ucb_score(self):
        """Test UCB score calculation."""
        parent = Node(prior=1.0)
        child = Node(prior=0.5)
        parent.visit_count = 10
        child.visit_count = 3
        child.value_sum = 2.0  # Value of 2/3
        
        ucb_score = self.mcts.ucb_score(parent, child)
        self.assertGreater(ucb_score, 0)
        
    def test_expand_node(self):
        """Test node expansion with network output."""
        node = Node(prior=1.0)
        mock_output = MockNetworkOutput(
            hidden_state=np.zeros(8),
            reward=0.0,
            policy_logits=[0.0, 1.0, -1.0],  # Three possible actions
            value=0.5
        )
        
        self.mcts.expand_node(node, to_play=1, network_output=mock_output)
        
        self.assertEqual(len(node.children), 3)
        self.assertTrue(node.expanded())
        self.assertEqual(node.to_play, 1)

    def test_backpropagate(self):
        """Test value backpropagation through the tree."""
        root = Node(prior=1.0)
        child1 = Node(prior=0.5)
        child2 = Node(prior=0.5)
        
        # Set up the to_play values to simulate alternating players
        root.to_play = 1    # Player 1's turn
        child1.to_play = -1  # Player -1's turn
        child2.to_play = 1   # Player 1's turn
        
        root.children[0] = child1
        child1.children[0] = child2
        
        search_path = [root, child1, child2]
        self.mcts.backpropagate(
            search_path=search_path,
            value=1.0,
            to_play=1,  # From perspective of player 1
            discount=1.0
        )
        
        # Check visit counts
        self.assertEqual(root.visit_count, 1)
        self.assertEqual(child1.visit_count, 1)
        self.assertEqual(child2.visit_count, 1)
        
        # Check values - should alternate based on player
        self.assertEqual(root.value_sum, 1.0)     # Player 1's node, positive
        self.assertEqual(child1.value_sum, -1.0)  # Player -1's node, negative
        self.assertEqual(child2.value_sum, 1.0)   # Player 1's node, positive

    def test_run_mcts(self):
        """Test complete MCTS run."""
        mock_network = MockNetwork(
            policy_logits=[0.0, 1.0, -1.0],
            value=0.5
        )
        root_state = np.zeros((8, 8))
        
        root = self.mcts.run(
            root_state=root_state,
            network=mock_network,
            to_play=1
        )
        
        self.assertEqual(root.visit_count, self.config.num_simulations)
        self.assertTrue(len(root.children) > 0)

    def test_dirichlet_noise(self):
        """Test Dirichlet noise is properly applied at root."""
        mock_network = MockNetwork(
            policy_logits=[0.0, 0.0, 0.0],  # Equal prior probabilities
            value=0.0
        )
        root_state = np.zeros((8, 8))
        
        # Run MCTS multiple times to check noise randomization
        roots = [
            self.mcts.run(root_state, mock_network, to_play=1)
            for _ in range(5)
        ]
        
        # Check that priors are different due to noise
        priors = [
            [child.prior for child in root.children.values()]
            for root in roots
        ]
        
        # Check that at least some priors are different
        self.assertTrue(any(
            np.any(np.array(p1) != np.array(p2))
            for i, p1 in enumerate(priors)
            for p2 in priors[i+1:]
        ))

class TestMinMaxStats(unittest.TestCase):
    def test_update_and_normalize(self):
        stats = MinMaxStats()
        
        # Test initialization
        self.assertEqual(stats.minimum, float('inf'))
        self.assertEqual(stats.maximum, -float('inf'))
        
        # Test updates
        values = [-1.0, 0.5, 2.0]
        for v in values:
            stats.update(v)
            
        self.assertEqual(stats.minimum, -1.0)
        self.assertEqual(stats.maximum, 2.0)
        
        # Test normalization
        self.assertEqual(stats.normalize(0.5), 0.5)
        self.assertEqual(stats.normalize(-1.0), 0.0)
        self.assertEqual(stats.normalize(2.0), 1.0)

class TestDiceRollHandler(unittest.TestCase):
    def test_roll_initialization(self):
        """Test that the 21 unique rolls are properly initialized."""
        handler = DiceRollHandler()
        
        # Should have exactly 21 rolls
        self.assertEqual(len(handler.rolls), 21)
        
        # Check probabilities sum to 1
        total_prob = sum(roll.probability for roll in handler.rolls)
        self.assertAlmostEqual(total_prob, 1.0)
        
        # Count doubles and verify their probabilities
        doubles = [roll for roll in handler.rolls if len(roll.moves) == 4]
        self.assertEqual(len(doubles), 6)  # Should be 6 doubles
        for double in doubles:
            self.assertAlmostEqual(double.probability, 1/36)
            
        # Check non-doubles
        non_doubles = [roll for roll in handler.rolls if len(roll.moves) == 2]
        self.assertEqual(len(non_doubles), 15)  # Should be 15 non-doubles
        for non_double in non_doubles:
            self.assertAlmostEqual(non_double.probability, 2/36)
            
    def test_roll_distribution(self):
        """Test that rolls follow the expected probability distribution."""
        handler = DiceRollHandler()
        roll_counts = defaultdict(int)
        
        # Simulate many rolls
        n_rolls = 10000
        for _ in range(n_rolls):
            roll = handler.get_roll()
            # Use tuple of moves as key for counting
            key = tuple(sorted(roll.moves[:2]))  # Only first 2 for doubles
            roll_counts[key] += 1
            
        # Verify roll frequencies match probabilities
        for roll in handler.rolls:
            key = tuple(sorted(roll.moves[:2]))
            expected = n_rolls * roll.probability
            actual = roll_counts[key]
            # Allow 20% deviation due to randomness
            self.assertLess(abs(actual - expected) / expected, 0.2)

class TestBackgammonMCTS(unittest.TestCase):
    def setUp(self):
        config = MCTSConfig(num_simulations=100)
        self.mcts = BackgammonMCTS(config)
    
    def test_mcts_with_doubles(self):
        """Test MCTS behavior with double rolls."""
        node = Node(prior=1.0)
        mock_output = MockNetworkOutput(
            hidden_state=np.zeros(8),
            reward=0.0,
            policy_logits=[0.1] * 10,
            value=0.5
        )
        
        # Test expansion with doubles (e.g., 6-6)
        moves = [6, 6, 6, 6]
        self.mcts.expand_node(node, to_play=1, network_output=mock_output,
                            available_moves=moves, roll_prob=1/36)
                            
        # Should create a child for each available move
        self.assertEqual(len(node.children), 1)  # Only action 6 is available
        child = node.children[6]
        # Calculate expected probability: policy_logit_prob * roll_prob * 36 (to normalize)
        policy_probs = np.exp(mock_output.policy_logits) / sum(np.exp(mock_output.policy_logits))
        expected_prob = policy_probs[6]  # Get probability for move 6
        self.assertAlmostEqual(child.prior * 36, expected_prob)
    
    def test_mcts_with_regular_roll(self):
        """Test MCTS behavior with non-double rolls."""
        node = Node(prior=1.0)
        mock_output = MockNetworkOutput(
            hidden_state=np.zeros(8),
            reward=0.0,
            policy_logits=[0.1] * 10,
            value=0.5
        )
        
        # Test expansion with regular roll (e.g., 3-5)
        moves = [3, 5]
        self.mcts.expand_node(node, to_play=1, network_output=mock_output,
                            available_moves=moves, roll_prob=2/36)
                            
        # Should create children for both moves
        self.assertEqual(len(node.children), 2)
        self.assertIn(3, node.children)
        self.assertIn(5, node.children)
        
        # Check probability adjustment
        policy_probs = np.exp(mock_output.policy_logits) / sum(np.exp(mock_output.policy_logits))
        for move, child in node.children.items():
            expected_prob = policy_probs[move]  # Get probability for this move
            # For regular rolls, multiply by 36/2 to normalize (2/36 is original roll prob)
            self.assertAlmostEqual(child.prior * 36 / 2, expected_prob)

if __name__ == '__main__':
    unittest.main()