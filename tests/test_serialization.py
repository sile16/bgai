"""Tests for serialization utilities."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from distributed.serialization import (
    serialize_weights,
    deserialize_weights,
    serialize_experience,
    deserialize_experience,
    serialize_rewards,
    deserialize_rewards,
    batch_experiences_to_jax,
    experiences_to_numpy_batch,
    serialize_checkpoint_metadata,
    deserialize_checkpoint_metadata,
    serialize_warm_tree,
    deserialize_warm_tree,
    get_serialized_size,
    estimate_experience_size,
)


class TestWeightSerialization:
    """Tests for model weight serialization."""

    def test_serialize_simple_weights(self):
        """Test serialization of simple weight dict."""
        params = {
            'layer1': jnp.ones((10, 10)),
            'layer2': jnp.zeros((5,)),
        }

        data = serialize_weights(params)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize_simple_weights(self):
        """Test deserialization of simple weight dict."""
        params = {
            'layer1': jnp.ones((10, 10)),
            'layer2': jnp.zeros((5,)),
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        assert 'layer1' in restored
        assert 'layer2' in restored
        assert jnp.allclose(restored['layer1'], params['layer1'])
        assert jnp.allclose(restored['layer2'], params['layer2'])

    def test_serialize_nested_weights(self):
        """Test serialization of nested weight structure."""
        params = {
            'params': {
                'encoder': {
                    'dense1': {'kernel': jnp.ones((32, 64)), 'bias': jnp.zeros((64,))},
                    'dense2': {'kernel': jnp.ones((64, 32)), 'bias': jnp.zeros((32,))},
                },
                'decoder': {
                    'dense1': {'kernel': jnp.ones((32, 64)), 'bias': jnp.zeros((64,))},
                },
            }
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        # Check nested structure
        assert 'params' in restored
        assert 'encoder' in restored['params']
        assert 'decoder' in restored['params']
        assert jnp.allclose(
            restored['params']['encoder']['dense1']['kernel'],
            params['params']['encoder']['dense1']['kernel']
        )

    def test_roundtrip_preserves_dtype(self):
        """Test that roundtrip preserves array dtype."""
        params = {
            'float32': jnp.ones((10,), dtype=jnp.float32),
            'float16': jnp.ones((10,), dtype=jnp.float16),
            'int32': jnp.ones((10,), dtype=jnp.int32),
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        assert restored['float32'].dtype == jnp.float32
        assert restored['float16'].dtype == jnp.float16
        assert restored['int32'].dtype == jnp.int32

    def test_roundtrip_preserves_shape(self):
        """Test that roundtrip preserves array shapes."""
        params = {
            'vec': jnp.ones((100,)),
            'mat': jnp.ones((32, 64)),
            'tensor': jnp.ones((4, 8, 16)),
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        assert restored['vec'].shape == (100,)
        assert restored['mat'].shape == (32, 64)
        assert restored['tensor'].shape == (4, 8, 16)

    def test_large_weights(self):
        """Test serialization of larger weight tensors."""
        params = {
            'large': jnp.ones((1000, 1000)),
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        assert restored['large'].shape == (1000, 1000)
        assert jnp.allclose(restored['large'], params['large'])


class TestExperienceSerialization:
    """Tests for game experience serialization."""

    def test_serialize_experience(self, sample_experience):
        """Test serialization of a single experience."""
        data = serialize_experience(sample_experience)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize_experience(self, sample_experience):
        """Test deserialization of a single experience."""
        data = serialize_experience(sample_experience)
        restored = deserialize_experience(data)

        assert 'observation_nn' in restored
        assert 'policy_weights' in restored
        assert 'policy_mask' in restored
        assert 'cur_player_id' in restored
        assert 'reward' in restored

    def test_experience_roundtrip_values(self, sample_experience):
        """Test that experience values are preserved."""
        data = serialize_experience(sample_experience)
        restored = deserialize_experience(data)

        # Restored values are numpy arrays
        np.testing.assert_array_almost_equal(
            restored['observation_nn'],
            np.array(sample_experience['observation_nn'])
        )
        np.testing.assert_array_almost_equal(
            restored['policy_weights'],
            np.array(sample_experience['policy_weights'])
        )

    def test_experience_with_none_reward(self):
        """Test experience with None reward (incomplete episode)."""
        exp = {
            'observation_nn': jnp.zeros((34,)),
            'policy_weights': jnp.ones((156,)) / 156,
            'policy_mask': jnp.ones((156,), dtype=jnp.bool_),
            'cur_player_id': 0,
            'reward': None,
        }

        data = serialize_experience(exp)
        restored = deserialize_experience(data)

        assert restored['reward'] is None

    def test_experience_size(self, sample_experience):
        """Test serialized experience size is reasonable."""
        data = serialize_experience(sample_experience)

        # Should be roughly (34 + 156 + 156 + 1 + 2) * 4 bytes + overhead
        # ~1400 bytes + msgpack overhead
        assert len(data) < 3000  # Reasonable upper bound


class TestRewardsSerialization:
    """Tests for reward serialization."""

    def test_serialize_rewards(self):
        """Test reward serialization."""
        rewards = jnp.array([1.0, -1.0])
        data = serialize_rewards(rewards)
        assert isinstance(data, bytes)

    def test_deserialize_rewards(self):
        """Test reward deserialization."""
        rewards = jnp.array([1.0, -1.0])
        data = serialize_rewards(rewards)
        restored = deserialize_rewards(data)

        np.testing.assert_array_almost_equal(restored, [1.0, -1.0])

    def test_rewards_different_lengths(self):
        """Test rewards with different number of players."""
        for num_players in [2, 3, 4]:
            rewards = jnp.ones(num_players)
            data = serialize_rewards(rewards)
            restored = deserialize_rewards(data)
            assert len(restored) == num_players


class TestBatchConversion:
    """Tests for batch conversion utilities."""

    def test_batch_experiences_to_jax(self, sample_experience):
        """Test converting list of experiences to JAX batch."""
        # Create multiple experiences
        experiences = []
        for i in range(16):
            exp_data = serialize_experience(sample_experience)
            reward_data = serialize_rewards(sample_experience['reward'])
            experiences.append({
                'data': exp_data,
                'final_rewards': reward_data,
                'model_version': 1,
            })

        batch = batch_experiences_to_jax(experiences)

        assert 'observation_nn' in batch
        assert 'policy_weights' in batch
        assert batch['observation_nn'].shape == (16, 34)
        assert batch['policy_weights'].shape == (16, 156)
        assert isinstance(batch['observation_nn'], jax.Array)

    def test_experiences_to_numpy_batch(self, sample_experience):
        """Test converting experiences to numpy batch."""
        experiences = []
        for i in range(8):
            exp_data = serialize_experience(sample_experience)
            reward_data = serialize_rewards(sample_experience['reward'])
            experiences.append({
                'data': exp_data,
                'final_rewards': reward_data,
            })

        batch = experiences_to_numpy_batch(experiences)

        assert 'observation_nn' in batch
        assert batch['observation_nn'].shape == (8, 34)
        assert isinstance(batch['observation_nn'], np.ndarray)

    def test_empty_batch(self):
        """Test handling of empty experience list."""
        batch = batch_experiences_to_jax([])
        assert batch == {}

    def test_batch_with_raw_experiences(self, sample_experience):
        """Test batch conversion with pre-deserialized experiences."""
        experiences = []
        for i in range(4):
            exp = {
                'data': {
                    'observation_nn': np.zeros((34,)),
                    'policy_weights': np.ones((156,)) / 156,
                },
                'final_rewards': np.array([1.0, -1.0]),
            }
            experiences.append(exp)

        batch = batch_experiences_to_jax(experiences)
        assert batch['observation_nn'].shape == (4, 34)


class TestCheckpointMetadata:
    """Tests for checkpoint metadata serialization."""

    def test_serialize_metadata(self):
        """Test metadata serialization."""
        metadata = {
            'step': 1000,
            'model_version': 5,
            'timestamp': 1234567890.123,
            'worker_id': 'trainer-001',
        }

        data = serialize_checkpoint_metadata(metadata)
        assert isinstance(data, bytes)

    def test_deserialize_metadata(self):
        """Test metadata deserialization."""
        metadata = {
            'step': 1000,
            'model_version': 5,
            'timestamp': 1234567890.123,
            'worker_id': 'trainer-001',
        }

        data = serialize_checkpoint_metadata(metadata)
        restored = deserialize_checkpoint_metadata(data)

        assert restored['step'] == 1000
        assert restored['model_version'] == 5
        assert restored['worker_id'] == 'trainer-001'


class TestUtilities:
    """Tests for utility functions."""

    def test_get_serialized_size_bytes(self):
        """Test size formatting for small data."""
        data = b'x' * 100
        size_str = get_serialized_size(data)
        assert 'B' in size_str
        assert '100' in size_str

    def test_get_serialized_size_kb(self):
        """Test size formatting for KB data."""
        data = b'x' * 2048
        size_str = get_serialized_size(data)
        assert 'KB' in size_str

    def test_get_serialized_size_mb(self):
        """Test size formatting for MB data."""
        data = b'x' * (2 * 1024 * 1024)
        size_str = get_serialized_size(data)
        assert 'MB' in size_str

    def test_estimate_experience_size(self):
        """Test experience size estimation."""
        size = estimate_experience_size()

        # Default backgammon: 34 obs, 156 actions, 2 players
        # Should be roughly 800-1200 bytes
        assert 500 < size < 2000

    def test_estimate_experience_size_custom(self):
        """Test experience size estimation with custom params."""
        size = estimate_experience_size(
            obs_dim=100,
            num_actions=300,
            num_players=4
        )

        # Larger observation and action space
        assert size > estimate_experience_size()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_serialize_empty_dict(self):
        """Test serialization of empty params dict."""
        params = {}
        data = serialize_weights(params)
        restored = deserialize_weights(data)
        assert restored == {}

    def test_serialize_scalar_values(self):
        """Test serialization with scalar values."""
        params = {
            'scalar': jnp.array(1.0),
            'int_scalar': jnp.array(42),
        }

        data = serialize_weights(params)
        restored = deserialize_weights(data)

        assert jnp.isclose(restored['scalar'], 1.0)
        assert restored['int_scalar'] == 42

    def test_experience_with_extra_fields(self):
        """Test experience with additional custom fields."""
        exp = {
            'observation_nn': jnp.zeros((34,)),
            'policy_weights': jnp.ones((156,)) / 156,
            'custom_field': 'some_value',
            'another_field': 123,
        }

        data = serialize_experience(exp)
        restored = deserialize_experience(data)  # Fixed: was passing exp instead of data

        assert restored['custom_field'] == 'some_value'
        assert restored['another_field'] == 123


class TestWarmTreeSerialization:
    """Tests for warm tree serialization."""

    def test_serialize_simple_tree_structure(self):
        """Test serialization of a simple tree-like structure."""
        # Create a mock tree structure similar to MCTSTree
        tree = {
            'data': {
                'n': jnp.zeros((100,)),
                'q': jnp.zeros((100,)),
                'p': jnp.zeros((100, 156)),
                'terminated': jnp.zeros((100,), dtype=jnp.bool_),
            },
            'edge_map': jnp.zeros((100, 156), dtype=jnp.int32),
            'parents': jnp.zeros((100,), dtype=jnp.int32),
            'next_free_idx': jnp.array(1),
        }

        data = serialize_warm_tree(tree)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize_simple_tree_structure(self):
        """Test deserialization of a tree structure."""
        tree = {
            'data': {
                'n': jnp.ones((50,)) * 10,
                'q': jnp.ones((50,)) * 0.5,
            },
            'edge_map': jnp.ones((50, 156), dtype=jnp.int32),
            'next_free_idx': jnp.array(25),
        }

        data = serialize_warm_tree(tree)
        restored = deserialize_warm_tree(data)

        assert 'data' in restored
        assert 'edge_map' in restored
        assert jnp.allclose(restored['data']['n'], tree['data']['n'])
        assert jnp.allclose(restored['data']['q'], tree['data']['q'])
        assert jnp.allclose(restored['next_free_idx'], jnp.array(25))

    def test_warm_tree_roundtrip_preserves_dtype(self):
        """Test that roundtrip preserves array dtypes."""
        tree = {
            'float_data': jnp.ones((10,), dtype=jnp.float32),
            'int_data': jnp.ones((10,), dtype=jnp.int32),
            'bool_data': jnp.ones((10,), dtype=jnp.bool_),
        }

        data = serialize_warm_tree(tree)
        restored = deserialize_warm_tree(data)

        assert restored['float_data'].dtype == jnp.float32
        assert restored['int_data'].dtype == jnp.int32
        assert restored['bool_data'].dtype == jnp.bool_

    def test_warm_tree_with_nested_structure(self):
        """Test tree with nested embedding structure."""
        tree = {
            'data': {
                'embedding': {
                    'observation': jnp.zeros((100, 34)),
                    'current_player': jnp.zeros((100,), dtype=jnp.int32),
                    '_is_stochastic': jnp.zeros((100,), dtype=jnp.bool_),
                },
                'n': jnp.zeros((100,)),
                'q': jnp.zeros((100,)),
            },
        }

        data = serialize_warm_tree(tree)
        restored = deserialize_warm_tree(data)

        assert 'embedding' in restored['data']
        assert restored['data']['embedding']['observation'].shape == (100, 34)
        assert restored['data']['embedding']['current_player'].dtype == jnp.int32

    def test_warm_tree_size_reasonable(self):
        """Test that serialized tree size is reasonable."""
        # Create a moderately sized tree
        max_nodes = 1000
        num_actions = 156
        obs_dim = 34

        tree = {
            'data': {
                'n': jnp.zeros((max_nodes,)),
                'q': jnp.zeros((max_nodes,)),
                'p': jnp.zeros((max_nodes, num_actions)),
                'terminated': jnp.zeros((max_nodes,), dtype=jnp.bool_),
                'embedding': {
                    'observation': jnp.zeros((max_nodes, obs_dim)),
                    'current_player': jnp.zeros((max_nodes,), dtype=jnp.int32),
                },
            },
            'edge_map': jnp.zeros((max_nodes, num_actions), dtype=jnp.int32),
            'parents': jnp.zeros((max_nodes,), dtype=jnp.int32),
        }

        data = serialize_warm_tree(tree)

        # Should be on the order of a few MB for 1000 nodes
        size_mb = len(data) / (1024 * 1024)
        assert size_mb < 50  # Reasonable upper bound
