import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

@dataclass
class NetworkOutput:
    """Output from the network containing value, reward, policy logits and hidden state."""
    value: float
    reward: float
    policy_logits: torch.Tensor
    hidden_state: Optional[torch.Tensor]

class BackgammonNetwork(nn.Module):
    """Neural network for backgammon value and policy prediction."""
    
    def __init__(self):
        super().__init__()
        # Input: 30 channels x 24 positions (from observation_shape)
        
        # Shared representation layers
        self.conv1 = nn.Conv1d(30, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # Policy head - outputs logits for all possible moves
        self.policy_conv = nn.Conv1d(128, 64, kernel_size=1)
        self.policy_fc = nn.Linear(64 * 24, 7128)  # 7128 possible moves
        
        # Value head - predicts game outcome
        self.value_conv = nn.Conv1d(128, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 24, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shared representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(-1, 64 * 24)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(-1, 32 * 24)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

    def initial_inference(self, state: torch.Tensor) -> NetworkOutput:
        """Initial model inference at root state."""
        policy_logits, value = self.forward(state)
        return NetworkOutput(
            value=value.item(),
            reward=0.0,
            policy_logits=policy_logits,
            hidden_state=None
        )

    def recurrent_inference(self, hidden_state: Optional[torch.Tensor], action: int) -> NetworkOutput:
        """Recurrent model inference for future states."""
        # For the base network, we don't use hidden states or actions
        # This is implemented fully in the MuZero variant
        policy_logits, value = self.forward(hidden_state)
        return NetworkOutput(
            value=value.item(),
            reward=0.0,
            policy_logits=policy_logits,
            hidden_state=None
        )

class BackgammonMuZeroNetwork(nn.Module):
    """MuZero-style network with dynamics and reward prediction."""
    
    def __init__(self):
        super().__init__()
        self.representation = BackgammonRepresentationNetwork()
        self.dynamics = BackgammonDynamicsNetwork()
        self.prediction = BackgammonPredictionNetwork()
        
    def initial_inference(self, state: torch.Tensor) -> NetworkOutput:
        """Initial model inference at root state."""
        hidden_state = self.representation(state)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(
            value=value.item(),
            reward=0.0,
            policy_logits=policy_logits,
            hidden_state=hidden_state
        )
        
    def recurrent_inference(self, hidden_state: torch.Tensor, action: int) -> NetworkOutput:
        """Recurrent model inference for future states."""
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return NetworkOutput(
            value=value.item(),
            reward=reward.item(),
            policy_logits=policy_logits,
            hidden_state=next_hidden_state
        )

class BackgammonRepresentationNetwork(nn.Module):
    """Encodes raw board state into abstract hidden state."""
    
    def __init__(self):
        super().__init__()
        # Similar architecture to main network's shared layers
        self.conv1 = nn.Conv1d(30, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class BackgammonDynamicsNetwork(nn.Module):
    """Predicts next hidden state and reward given current hidden state and action."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(129, 128, kernel_size=3, padding=1)  # 128 + 1 for action
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # Reward prediction
        self.reward_conv = nn.Conv1d(128, 32, kernel_size=1)
        self.reward_fc = nn.Linear(32 * 24, 1)
        
    def forward(self, hidden_state, action):
        # Encode action and concatenate with hidden state
        batch_size = hidden_state.shape[0]
        action_plane = torch.full((batch_size, 1, 24), action, 
                                dtype=hidden_state.dtype, 
                                device=hidden_state.device)
        x = torch.cat([hidden_state, action_plane], dim=1)
        
        # Predict next state
        x = F.relu(self.conv1(x))
        next_hidden_state = F.relu(self.conv2(x))
        
        # Predict reward
        reward = F.relu(self.reward_conv(next_hidden_state))
        reward = reward.view(batch_size, -1)
        reward = torch.tanh(self.reward_fc(reward))
        
        return next_hidden_state, reward

class BackgammonPredictionNetwork(nn.Module):
    """Predicts policy and value from hidden state."""
    
    def __init__(self):
        super().__init__()
        # Similar to policy and value heads of main network
        self.policy_conv = nn.Conv1d(128, 64, kernel_size=1)
        self.policy_fc = nn.Linear(64 * 24, 7128)
        
        self.value_conv = nn.Conv1d(128, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 24, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, hidden_state):
        # Policy head
        policy = F.relu(self.policy_conv(hidden_state))
        policy = policy.view(-1, 64 * 24)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_conv(hidden_state))
        value = value.view(-1, 32 * 24)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value