import pytest
import numpy as np
from mcts import MCTS, Node, MCTSConfig
from src.backgammon_env import BackgammonEnv

class MockNeuralNet:
    """Mock neural network for testing MCTS."""
    def __init__(self, policy_output=None, value_output=0.0):
        self.policy_output = policy_output
        self.value_output = value_output
        self.eval_count = 0
    
    def evaluate(self, board):
        self.eval_count += 1
        if self.policy_output is None:
            # Generate uniform policy over legal moves
            moves = board.get_legal_moves()
            self.policy_output = [1.0 / len(moves)] * len(moves)
        return self.policy_output, self.value_output

class TestMCTS:
    @pytest.fixture
    def env(self):
        return BackgammonEnv()
    
    @pytest.fixture
    def config(self):
        return MCTSConfig(
            num_simulations=100,
            cpuct=1.0,
            temperature=1.0,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25
        )
    
    def test_mcts_initialization(self, env, config):
        net = MockNeuralNet()
        mcts = MCTS(env.get_state(), config, net)
        
        assert mcts.root is not None
        assert mcts.root.parent is None
        assert len(mcts.root.children) == 0
        assert mcts.root.visits == 0
        assert mcts.root.total_value == 0.0
    
    def test_single_simulation(self, env, config):
        net = MockNeuralNet(value_output=0.5)
        mcts = MCTS(env.get_state(), config, net)
        
        mcts.run_simulation()
        
        assert mcts.root.visits == 1
        assert len(mcts.root.children) > 0
        assert net.eval_count == 1
    
    def test_multiple_simulations(self, env, config):
        net = MockNeuralNet()
        mcts = MCTS(env.get_state(), config, net)
        
        num_sims = 10
        for _ in range(num_sims):
            mcts.run_simulation()
        
        assert mcts.root.visits == num_sims
        assert net.eval_count <= num_sims  # May be less due to cache hits
    
    def test_puct_selection(self, env, config):
        """Test that PUCT formula correctly balances exploration/exploitation."""
        net = MockNeuralNet(value_output=0.0)
        mcts = MCTS(env.get_state(), config, net)
        
        # Run enough simulations to build statistics
        for _ in range(50):
            mcts.run_simulation()
        
        # Get action probabilities
        probs = mcts.get_action_probs(temperature=1.0)
        
        # Check that probabilities sum to 1
        assert np.abs(np.sum(probs) - 1.0) < 1e-6
        # Check that all probabilities are non-negative
        assert np.all(probs >= 0)
    
    def test_temperature_scaling(self, env, config):
        """Test that temperature parameter affects action distribution."""
        net = MockNeuralNet()
        mcts = MCTS(env.get_state(), config, net)
        
        # Run simulations
        for _ in range(50):
            mcts.run_simulation()
        
        # Compare distributions at different temperatures
        probs_high_temp = mcts.get_action_probs(temperature=1.0)
        probs_low_temp = mcts.get_action_probs(temperature=0.1)
        
        # Lower temperature should give more extreme distribution
        assert np.max(probs_low_temp) > np.max(probs_high_temp)
        assert np.min(probs_low_temp) < np.min(probs_high_temp)
    
    def test_dirichlet_noise(self, env, config):
        """Test that Dirichlet noise affects root node policy."""
        net = MockNeuralNet()
        mcts = MCTS(env.get_state(), config, net)
        
        # Run without noise
        config.dirichlet_eps = 0.0
        mcts1 = MCTS(env.get_state(), config, net)
        probs1 = mcts1.get_action_probs(temperature=1.0)
        
        # Run with noise
        config.dirichlet_eps = 0.25
        mcts2 = MCTS(env.get_state(), config, net)
        probs2 = mcts2.get_action_probs(temperature=1.0)
        
        # Distributions should be different
        assert not np.allclose(probs1, probs2)
    
    def test_game_result_backup(self, env, config):
        """Test that game results are correctly backed up through the tree."""
        net = MockNeuralNet(value_output=1.0)  # Always predict win
        mcts = MCTS(env.get_state(), config, net)
        
        # Run simulation to build tree
        mcts.run_simulation()
        
        # Check that values are correctly propagated
        def check_values(node):
            if len(node.children) == 0:
                return
            for child in node.children.values():
                assert child.total_value <= node.visits  # Values should be bounded
                check_values(child)
        
        check_values(mcts.root)

    def test_parallel_safety(self, env, config):
        """Test that MCTS is thread-safe."""
        import threading
        
        net = MockNeuralNet()
        mcts = MCTS(env.get_state(), config, net)
        
        def run_simulations():
            for _ in range(10):
                mcts.run_simulation()
        
        # Run simulations in parallel
        threads = []
        for _ in range(4):
            t = threading.Thread(target=run_simulations)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check that tree is consistent
        assert mcts.root.visits == 40  # 4 threads × 10 simulations each
