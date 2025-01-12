import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

class PerformanceAnalyzer:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.metrics_history = []
        
    def analyze_training_run(self, run_id):
        """Analyze a complete training run."""
        metrics_file = self.log_dir / f'metrics_{run_id}.json'
        prof_file = self.log_dir / f'profile_{run_id}.pt'
        
        # Load and analyze metrics
        df = pd.read_json(metrics_file)
        
        # Calculate key statistics
        stats = {
            'avg_gpu_util': df['gpu_util'].mean(),
            'avg_gpu_mem': df['gpu_memory'].mean(),
            'avg_batch_time': df['batch_time'].mean(),
            'throughput': len(df) * df['batch_size'].mean() / df['total_time'].sum()
        }
        
        # Load profiler results
        if prof_file.exists():
            prof_data = torch.load(prof_file)
            stats.update(self._analyze_profile(prof_data))
        
        return stats
    
    def _analyze_profile(self, prof_data):
        """Analyze PyTorch profiler data."""
        stats = {}
        
        # Compute kernel statistics
        kernel_times = []
        memory_usage = []
        
        for event in prof_data.events():
            if event.kind == 'cuda':
                kernel_times.append(event.duration)
                memory_usage.append(event.memory_usage)
        
        stats['kernel_time_mean'] = np.mean(kernel_times)
        stats['kernel_time_std'] = np.std(kernel_times)
        stats['peak_memory'] = max(memory_usage)
        
        return stats
    
    def plot_training_metrics(self, run_id):
        """Generate performance visualization plots."""
        metrics_file = self.log_dir / f'metrics_{run_id}.json'
        df = pd.read_json(metrics_file)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU Utilization
        sns.lineplot(data=df, x='step', y='gpu_util', ax=axes[0,0])
        axes[0,0].set_title('GPU Utilization')
        
        # Memory Usage
        sns.lineplot(data=df, x='step', y='gpu_memory', ax=axes[0,1])
        axes[0,1].set_title('GPU Memory Usage')
        
        # Batch Time
        sns.histplot(data=df, x='batch_time', ax=axes[1,0])
        axes[1,0].set_title('Batch Time Distribution')
        
        # Loss
        sns.lineplot(data=df, x='step', y='loss', ax=axes[1,1])
        axes[1,1].set_title('Training Loss')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, run_id):
        """Generate a comprehensive performance report."""
        stats = self.analyze_training_run(run_id)
        fig = self.plot_training_metrics(run_id)
        
        report = {
            'run_id': run_id,
            'performance_metrics': stats,
            'bottlenecks': self._identify_bottlenecks(stats),
            'recommendations': self._generate_recommendations(stats)
        }
        
        return report
    
    def _identify_bottlenecks(self, stats):
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if stats['avg_gpu_util'] < 80:
            bottlenecks.append({
                'type': 'GPU Utilization',
                'severity': 'high',
                'description': 'GPU is underutilized, possibly due to data loading or preprocessing bottlenecks'
            })
        
        if stats['avg_batch_time'] > 0.1:  # 100ms threshold
            bottlenecks.append({
                'type': 'Batch Processing',
                'severity': 'medium',
                'description': 'Batch processing time is high, consider optimizing model or reducing batch size'
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, stats):
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if stats['avg_gpu_util'] < 80:
            recommendations.append({
                'category': 'Data Pipeline',
                'suggestion': 'Increase number of worker processes and prefetch factor',
                'priority': 'high'
            })
        
        if stats['peak_memory'] / 1e9 < 20:  # Less than 20GB on RTX 4090
            recommendations.append({
                'category': 'Memory Usage',
                'suggestion': 'Increase batch size to utilize more GPU memory',
                'priority': 'medium'
            })
        
        return recommendations