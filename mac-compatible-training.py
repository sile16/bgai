import torch
import platform

class DevTrainer:
    """Simplified trainer for development and testing."""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Detect platform and set device
        if platform.system() == "Darwin":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.use_amp = False  # MPS doesn't support automatic mixed precision
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.use_amp = True
        
        self.model = self.model.to(self.device)
        
        # Simpler optimizer setup for development
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )

    def train_step(self, batch):
        states, policies, values = [b.to(self.device) for b in batch]
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                policy_pred, value_pred = self.model(states)
                loss = self._compute_loss(policy_pred, value_pred, policies, values)
        else:
            policy_pred, value_pred = self.model(states)
            loss = self._compute_loss(policy_pred, value_pred, policies, values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'device': self.device.type
        }

    def _compute_loss(self, policy_pred, value_pred, policy_target, value_target):
        policy_loss = torch.nn.functional.cross_entropy(policy_pred, policy_target)
        value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_target)
        return policy_loss + value_loss

def train_development(model, train_loader, config):
    """Simple training loop for development and testing."""
    trainer = DevTrainer(model, config)
    
    print(f"Training on device: {trainer.device}")
    
    for epoch in range(config['num_epochs']):
        for batch_idx, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            
            if batch_idx % config['log_interval'] == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['loss']:.4f}")
            
            # Early exit for testing
            if config.get('test_mode') and batch_idx >= 5:
                return
