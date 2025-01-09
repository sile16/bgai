import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BackgammonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_input_features = 196  # From our enhanced state encoding
        self.num_actions = 256  # Upper bound on possible moves
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(self.num_input_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, x):
        # x shape: [batch_size, num_input_features]
        shared_output = self.shared(x)
        policy = self.policy_head(shared_output)
        value = self.value_head(shared_output)
        return policy, value

class BackgammonDataset(Dataset):
    def __init__(self, states, policy_targets, value_targets):
        self.states = torch.FloatTensor(states)
        self.policy_targets = torch.LongTensor(policy_targets)
        self.value_targets = torch.FloatTensor(value_targets)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policy_targets[idx], self.value_targets[idx]

def train_batch(model, optimizer, batch, device):
    states, policy_targets, value_targets = [b.to(device) for b in batch]
    
    # Forward pass
    policy_logits, value_preds = model(states)
    
    # Calculate losses
    policy_loss = F.cross_entropy(policy_logits, policy_targets)
    value_loss = F.mse_loss(value_preds.squeeze(-1), value_targets)
    
    # Combined loss
    loss = policy_loss + value_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': loss.item()
    }

def train(model, train_loader, num_epochs, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            losses = train_batch(model, optimizer, batch, device)
            epoch_losses.append(losses)
            
        scheduler.step()
        
        # Log metrics, save checkpoints, etc.
        print(f"Epoch {epoch}: Loss = {np.mean([l['total_loss'] for l in epoch_losses]):.4f}")

def setup_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    model = BackgammonNet()
    batch_size = 1024  # Adjust based on GPU memory
    
    return model, device, batch_size