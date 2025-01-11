import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import psutil
import GPUtil

def test_gpu():
    """Test GPU availability and performance."""
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Basic GPU tensor operations
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        z = torch.matmul(x, y)
        end.record()
        torch.cuda.synchronize()
        
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f}ms")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    else:
        print("\nNo GPU available. Using CPU only.")

def test_dependencies():
    """Test key dependencies."""
    print("\nDependency Versions:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # Test matplotlib
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.close()
    print("Matplotlib: Working")
    
    # Test system monitoring
    print("\nSystem Information:")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    
    if GPUtil.getGPUs():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU Load: {gpu.load*100:.1f}%")
        print(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

def main():
    print("Testing environment setup...")
    
    test_dependencies()
    test_gpu()
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    main()