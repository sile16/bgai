import subprocess
import sys
import platform
from pathlib import Path

def get_cuda_torch_version():
    """Determine the correct torch version based on system."""
    system = platform.system()
    if system == "Darwin":
        # MacOS uses MPS (Metal Performance Shaders)
        return "torch torchvision"
    else:
        # For systems with NVIDIA GPUs
        return "torch torchvision --index-url https://download.pytorch.org/whl/cu121"

def main():
    # Ensure pip is up to date
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install PyTorch with appropriate CUDA version
    torch_packages = get_cuda_torch_version()
    subprocess.run([sys.executable, "-m", "pip", "install", *torch_packages.split()])
    
    # Install other requirements
    requirements_path = Path(__file__).parent / "requirements.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    
    # Verify torch installation
    try:
        import torch
        print("\nPyTorch installation verified:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("Failed to import torch. Installation may have failed.")

if __name__ == "__main__":
    main()