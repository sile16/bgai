import torch
import torch.cuda.amp
import torch.profiler
import torch.nn.functional as F
import numpy as np
import time
import psutil
import GPUtil
from pathlib import Path
from datetime import datetime
import logging
import wandb  # For experiment tracking
from torch.utils.data import DataLoader
import threading
import queue

class GPUMonitor:
    def __init__(self, gpu_id=0, log_interval=1):
        self.gpu_id = gpu_id
        self.log_interval = log_interval
        self.running = True
        self.stats_queue = queue.Queue()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def _monitor_loop(self):
        while self.running:
            gpu = GPUtil.getGPUs()[self.gpu_id]
            cpu_percent = psutil.cpu_percent()
            ram_percent = psutil.virtual_memory().percent
            
            stats = {
                'gpu_util': gpu.load * 100,
                'gpu_memory': gpu.memoryUtil * 100,
                'gpu_temp': gpu.temperature,
                'cpu_util': cpu_percent,
                'ram_util': ram_percent
            }
            self.stats_queue.put(stats)
            time.sleep(self.log_interval)

    def get_stats(self):
        if not self.stats_queue.empty():
            return self.stats_queue.get()
        return None

    def stop(self):
        self.running = False
        self.thread.join()

class TrainingManager:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda')
        self.scaler = torch.cuda.amp.GradScaler()
        self.monitor = GPUMonitor()
        
        # Optimize for RTX 4090
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Initialize optimizer with learning rate warmup
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.1,
            anneal_strategy='cos'
        )

    def train_step(self, batch, batch_idx):
        states, policies, values = [b.to(self.device) for b in batch]
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            policy_pred, value_pred = self.model(states)
            policy_loss = F.cross_entropy(policy_pred, policies)
            value_loss = F.mse_loss(value_pred.squeeze(), values)
            loss = policy_loss + value_loss

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

def train(config):
    # Initialize wandb for experiment tracking
    wandb.init(project="backgammon-ai", config=config)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create model and move to GPU
    model = BackgammonNet()
    model = torch.compile(model)  # Use torch.compile for extra speed
    model = model.to('cuda')
    
    # Initialize training manager
    trainer = TrainingManager(model, config)
    
    # Setup profiler
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    # Training loop
    step = 0
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            with profiler:
                metrics = trainer.train_step(batch, batch_idx)
                
                # Get hardware stats
                hw_stats = trainer.monitor.get_stats()
                if hw_stats:
                    metrics.update(hw_stats)
                
                # Log metrics
                wandb.log(metrics)
                
                if batch_idx % config['log_interval'] == 0:
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                    logger.info(
                        f"Epoch {epoch}/{config['num_epochs']} "
                        f"[{batch_idx}/{len(train_loader)}] "
                        f"Loss: {metrics['loss']:.4f} "
                        f"GPU Mem: {gpu_mem:.1f}GB"
                    )
                
                step += 1
            
            profiler.step()
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'loss': metrics['loss'],
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

if __name__ == "__main__":
    config = {
        'batch_size': 2048,  # Optimized for RTX 4090 24GB
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'steps_per_epoch': 1000,
        'log_interval': 10,
        'save_interval': 5,
        'num_workers': 8,  # For data loading
    }
    
    # Initialize data pipeline
    data_loader = ContinuousDataLoader(
        data_dir='game_data',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        prefetch_factor=2
    )

    train(config)