#!/usr/bin/env python3
"""
Common functionality for benchmark scripts.
"""

import os
import sys
import time
import json
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
DEFAULT_MEMORY_LIMIT_GB = 24  # Maximum memory to use (in GB)
DEFAULT_BENCHMARK_DURATION = 30  # Duration of each batch size test in seconds

# Get the directory where the benchmark script is located
BENCHMARK_DIR = Path(__file__).parent.absolute()
PROFILE_DIR = BENCHMARK_DIR / "profiles"
GRAPHS_DIR = BENCHMARK_DIR / "graphs"
HUMAN_READABLE_UNITS = ["", "K", "M", "B", "T"]


def convert_to_python(value):
    """Convert JAX arrays and other non-JSON-serializable types to Python native types."""
    if hasattr(value, 'item'):  # JAX/numpy scalar arrays
        return value.item()
    if isinstance(value, (jnp.ndarray, jax.Array)):
        if value.ndim == 0:
            return float(value)
        return [convert_to_python(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value)
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [convert_to_python(v) for v in value]
    if isinstance(value, dict):
        return {k: convert_to_python(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return float(value) if isinstance(value, np.floating) else int(value)
    return value


# Define common data structures
class BatchBenchResult(NamedTuple):
    batch_size: int
    steps_per_second: float  # Renamed from moves_per_second for consistency
    games_per_second: float
    avg_game_length: float
    median_game_length: float
    min_game_length: float
    max_game_length: float
    memory_usage_gb: float
    memory_usage_percent: float  # New field for memory percentage
    efficiency: float
    valid: bool = True
    
    # For backward compatibility, provide moves_per_second as alias
    @property
    def moves_per_second(self) -> float:
        return self.steps_per_second


@dataclass
class BenchmarkProfile:
    """Profile containing hardware information and benchmark results."""
    # Hardware info
    platform: str
    processor: str
    jaxlib_type: str  # cpu, cuda, metal
    device_info: str
    python_version: str
    jax_version: str
    
    # Enhanced hardware identification
    device_name: str = "unknown"  # Specific device name (e.g., "rtx-4090", "arm-cpu")
    device_platform: str = "unknown"  # JAX device platform (cuda, metal, cpu)
    num_devices: int = 1  # Number of JAX devices
    
    # Benchmark results - standardized naming
    benchmark_name: str = "Unknown"  # Name of the benchmark (e.g., "MCTS", "StochasticMCTS", "GameEnv")
    batch_sizes: List[int] = field(default_factory=list)
    steps_per_second: List[float] = field(default_factory=list)  # Primary performance metric
    memory_usage_gb: List[float] = field(default_factory=list)
    memory_usage_percent: List[float] = field(default_factory=list)  # Memory usage as percentage of total available
    timestamp: str = None  # Creation date
    duration: int = DEFAULT_BENCHMARK_DURATION  # Duration used for each benchmark in seconds
    
    # Optional metrics
    games_per_second: List[float] = None  
    efficiency: List[float] = None  # Steps per second per GB of memory
    extra_info: Dict[str, Any] = None  # Additional benchmark-specific data
    
    # For backward compatibility
    moves_per_second: List[float] = None
    
    def __post_init__(self):
        # Set timestamp if none provided
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


def format_human_readable(num: float) -> str:
    """Format a number in human-readable form with appropriate units."""
    idx = 0
    while abs(num) >= 1000 and idx < len(HUMAN_READABLE_UNITS) - 1:
        num /= 1000
        idx += 1
    return f"{num:.2f}{HUMAN_READABLE_UNITS[idx]}"


def get_version_info() -> Dict[str, str]:
    """Get version information for packages and git repos."""
    version_info = {}
    
    # Get PGX version
    try:
        import pgx
        version_info["pgx_version"] = pgx.__version__
    except ImportError:
        version_info["pgx_version"] = "not_installed"
    except AttributeError:
        version_info["pgx_version"] = "unknown"
    
    # Get TurboZero version info using pip
    try:
        import subprocess
        result = subprocess.run(
            ["pip", "show", "turbozero"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse pip show output
            turbozero_version = "unknown"
            install_location = "unknown"
            
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    turbozero_version = line.split(":", 1)[1].strip()
                elif line.startswith("Location:"):
                    install_location = line.split(":", 1)[1].strip()
            
            version_info["turbozero_version"] = turbozero_version
            
            # Try to get installation timestamp from the dist-info directory
            try:
                import os
                import stat
                from datetime import datetime
                
                # Look for dist-info directory
                dist_info_pattern = f"turbozero-{turbozero_version}.dist-info"
                dist_info_path = Path(install_location) / dist_info_pattern
                
                if dist_info_path.exists():
                    # Get modification time of the directory (install time)
                    mtime = os.path.getmtime(dist_info_path)
                    install_time = datetime.fromtimestamp(mtime).isoformat()
                    version_info["turbozero_install_time"] = install_time
                else:
                    version_info["turbozero_install_time"] = "unknown"
                    
            except Exception:
                version_info["turbozero_install_time"] = "unknown"
                
        else:
            version_info["turbozero_version"] = "not_installed"
            version_info["turbozero_install_time"] = "not_installed"
            
    except Exception:
        version_info["turbozero_version"] = "not_installed" 
        version_info["turbozero_install_time"] = "not_installed"
    
    return version_info


def get_system_info() -> Dict[str, str]:
    """Get system information for benchmarking context."""
    device = jax.devices()[0] if jax.devices() else None
    
    # Get detailed device information
    device_platform = device.platform if device else "unknown"
    device_kind = getattr(device, 'device_kind', 'unknown') if device else "unknown"
    device_type = device_platform
    
    # Use device_kind as the device name for better identification
    device_name = device_kind.lower().replace(" ", "-") if device_kind != "unknown" else "unknown"
    
    # For CPU devices, use processor info
    if device_platform == "cpu":
        if "arm" in platform.processor().lower():
            device_name = "arm-cpu"
        else:
            device_name = platform.processor().split()[0] if platform.processor() else "cpu"
    
    # Get CPU info
    cpu_count = os.cpu_count() or 1
    cpu_type = platform.machine()
    
    # Get version information
    version_info = get_version_info()
    
    base_info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_type": device_type,
        "device_info": str(device) if device else "unknown",
        "device_name": device_name,
        "device_kind": device_kind,
        "num_devices": len(jax.devices()),
        "device_platform": device_platform,
        "cpu_count": cpu_count,
        "cpu_type": cpu_type,
    }
    
    # Merge with version info
    base_info.update(version_info)
    return base_info


def print_system_info(system_info: Dict[str, str]) -> None:
    """Print system information to the console."""
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")


def get_memory_usage() -> float:
    """Get current memory usage in GB, attempting device-specific reporting."""
    sys_info = get_system_info()
    jaxlib_type = sys_info.get('jaxlib_type', 'cpu')
    device = jax.devices()[0] if jax.devices() else None
    
    # Static variable to track if we've printed the memory source message
    if not hasattr(get_memory_usage, "_printed_source"):
        get_memory_usage._printed_source = False
    
    # Helper function for process RSS
    def get_process_rss_gb():
        if platform.system() == "Darwin":
            try:
                cmd = "ps -o rss= -p " + str(os.getpid())
                output = subprocess.check_output(cmd.split(), timeout=1).decode().strip()
                mem_kb = float(output)
                return mem_kb / 1024 / 1024  # Convert KB to GB
            except Exception as e:
                print(f"Warning: Failed to get macOS process memory via ps: {e}", flush=True)
        elif platform.system() == "Linux":
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            mem_kb = float(line.split()[1])
                            return mem_kb / 1024 / 1024  # Convert KB to GB
            except Exception as e:
                print(f"Warning: Failed to get Linux process memory via /proc: {e}", flush=True)
        
        # Fallback using psutil if specific methods fail or not implemented
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss
            return mem_bytes / 1024 / 1024 / 1024  # Convert bytes to GB
        except ImportError:
            print("Warning: psutil not installed, cannot provide fallback memory usage.", flush=True)
            return 0.0
        except Exception as e:
            print(f"Warning: Failed to get memory via psutil: {e}", flush=True)
            return 0.0

    # --- Device-specific logic ---
    
    if jaxlib_type == 'cuda' and device is not None:
        try:
            # Use nvidia-smi to get GPU memory usage for the primary JAX device
            device_id = device.id
            cmd = [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                f"--format=csv,noheader,nounits",
                f"-i", f"{device_id}"
            ]
            output = subprocess.check_output(cmd, timeout=2).decode().strip()
            mem_mib = float(output)
            if not get_memory_usage._printed_source:
                print(f"Reporting CUDA device {device_id} memory via nvidia-smi.", flush=True)
                get_memory_usage._printed_source = True
            return mem_mib / 1024  # Convert MiB to GB
        except FileNotFoundError:
            print("Warning: nvidia-smi not found. Falling back to process RSS for memory usage.", flush=True)
            return get_process_rss_gb()
        except subprocess.TimeoutExpired:
            print("Warning: nvidia-smi timed out. Falling back to process RSS.", flush=True)
            return get_process_rss_gb()
        except Exception as e:
            print(f"Warning: Failed to get GPU memory via nvidia-smi: {e}. Falling back to process RSS.", flush=True)
            return get_process_rss_gb()
            
    elif jaxlib_type == 'metal':
        # For Metal (macOS unified memory), process RSS is the best available proxy
        if not get_memory_usage._printed_source:
            print("Reporting host process memory usage (psutil/ps) for Metal backend.", flush=True)
            get_memory_usage._printed_source = True
        return get_process_rss_gb()
        
    else: # cpu or unknown
        # For CPU backend, process RSS is appropriate
        if not get_memory_usage._printed_source:
            print(f"Reporting host process memory usage (psutil/ps/proc) for {jaxlib_type} backend.", flush=True)
            get_memory_usage._printed_source = True
        return get_process_rss_gb()


def get_total_memory_gb() -> float:
    """Get total available memory in GB for the current device type."""
    sys_info = get_system_info()
    jaxlib_type = sys_info.get('jaxlib_type', 'cpu')
    
    if jaxlib_type == 'cuda':
        try:
            device = jax.devices()[0]
            device_id = device.id
            cmd = [
                "nvidia-smi",
                f"--query-gpu=memory.total",
                f"--format=csv,noheader,nounits",
                f"-i", f"{device_id}"
            ]
            output = subprocess.check_output(cmd, timeout=2).decode().strip()
            mem_mib = float(output)
            return mem_mib / 1024  # Convert MiB to GB
        except Exception:
            pass
    
    # For CPU/Metal or fallback, use system RAM
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024 * 1024)
    except ImportError:
        # Conservative estimate if psutil not available
        return 16.0  # 16GB default

def calculate_memory_percentage(memory_gb: float) -> float:
    """Calculate memory usage as a percentage of total available memory."""
    total_memory = get_total_memory_gb()
    return (memory_gb / total_memory) * 100.0

def create_profile_filename(benchmark_name: str = None, **kwargs) -> str:
    """Create a unique filename for the profile based on system info and benchmark parameters."""
    sys_info = get_system_info()
    platform_name = sys_info["platform"].lower()
    device_name = sys_info["device_name"]
    device_platform = sys_info["device_platform"]
    num_devices = sys_info["num_devices"]
    
    # Create a unique identifier based on all relevant system information
    system_id_parts = [
        platform_name,
        device_name,
        device_platform,
        f"dev{num_devices}"
    ]
    
    system_id = "_".join(system_id_parts)
    
    if benchmark_name:
        benchmark_id = benchmark_name.lower()
        
        # Add MCTS-specific parameters to filename for unique identification
        if "num_simulations" in kwargs and "max_nodes" in kwargs:
            benchmark_id += f"_sims{kwargs['num_simulations']}_nodes{kwargs['max_nodes']}"
        
        filename = f"{system_id}_{benchmark_id}.json"
    else:
        filename = f"{system_id}.json"
    
    print(f"Profile filename: {filename}", flush=True)
    return filename


def profile_matches_system(profile_data: Dict[str, Any]) -> bool:
    """Check if a profile matches the current system configuration."""
    current_sys_info = get_system_info()
    
    # Check critical matching criteria
    match_criteria = [
        ("platform", "platform"),
        ("device_name", "device_name"),
        ("device_platform", "device_platform"),
        ("num_devices", "num_devices"),
    ]
    
    for profile_key, sys_key in match_criteria:
        profile_value = profile_data.get(profile_key)
        current_value = current_sys_info.get(sys_key)
        
        if profile_value != current_value:
            print(f"Profile mismatch: {profile_key} ({profile_value}) != current {sys_key} ({current_value})")
            return False
    
    return True


def find_matching_profile(benchmark_name: str, **kwargs) -> Optional[Path]:
    """Find a profile that matches the current system configuration for the given benchmark."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # First try exact filename match
    exact_filename = create_profile_filename(benchmark_name, **kwargs)
    exact_filepath = PROFILE_DIR / exact_filename
    
    if exact_filepath.exists():
        print(f"Found exact profile match: {exact_filename}")
        return exact_filepath
    
    # If no exact match, search through all profiles for this benchmark
    benchmark_id = benchmark_name.lower()
    if "num_simulations" in kwargs and "max_nodes" in kwargs:
        benchmark_id += f"_sims{kwargs['num_simulations']}_nodes{kwargs['max_nodes']}"
    
    all_profiles = list(PROFILE_DIR.glob(f"*_{benchmark_id}.json"))
    
    for profile_path in all_profiles:
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            if profile_matches_system(profile_data):
                print(f"Found matching profile: {profile_path.name}")
                return profile_path
                
        except Exception as e:
            print(f"Error reading profile {profile_path}: {e}")
            continue
    
    print(f"No matching profile found for {benchmark_name} with params {kwargs}")
    return None


def insert_batch_result_to_profile(batch_result: BatchBenchResult, benchmark_name: str, **kwargs) -> Optional[Path]:
    """Insert or update a single batch result in an existing profile."""
    # Try to load existing profile
    existing_profile = load_profile(benchmark_name, **kwargs)
    
    if not existing_profile:
        print(f"No existing profile found for {benchmark_name}. Cannot insert batch result.")
        return None
    
    batch_size = batch_result.batch_size
    print(f"Inserting/updating batch size {batch_size} in profile for {benchmark_name}")
    
    # Find the position to insert/update
    try:
        existing_idx = existing_profile.batch_sizes.index(batch_size)
        print(f"Updating existing batch size {batch_size} at index {existing_idx}")
        
        # Update existing entry
        existing_profile.steps_per_second[existing_idx] = batch_result.steps_per_second
        existing_profile.memory_usage_gb[existing_idx] = batch_result.memory_usage_gb
        existing_profile.memory_usage_percent[existing_idx] = batch_result.memory_usage_percent
        existing_profile.efficiency[existing_idx] = batch_result.efficiency
        
        if existing_profile.games_per_second is not None:
            existing_profile.games_per_second[existing_idx] = batch_result.games_per_second
            
    except ValueError:
        print(f"Adding new batch size {batch_size} to profile")
        
        # Find correct position to maintain sorted order
        insert_idx = 0
        for i, existing_batch in enumerate(existing_profile.batch_sizes):
            if batch_size < existing_batch:
                insert_idx = i
                break
            insert_idx = i + 1
        
        # Insert at correct position
        existing_profile.batch_sizes.insert(insert_idx, batch_size)
        existing_profile.steps_per_second.insert(insert_idx, batch_result.steps_per_second)
        existing_profile.memory_usage_gb.insert(insert_idx, batch_result.memory_usage_gb)
        existing_profile.memory_usage_percent.insert(insert_idx, batch_result.memory_usage_percent)
        existing_profile.efficiency.insert(insert_idx, batch_result.efficiency)
        
        if existing_profile.games_per_second is not None:
            existing_profile.games_per_second.insert(insert_idx, batch_result.games_per_second)
    
    # Update timestamp
    existing_profile.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the updated profile
    return save_profile(existing_profile, **kwargs)


def generate_graphs_from_profile(profile: BenchmarkProfile) -> Tuple[str, str]:
    """Generate benchmark graphs from a complete profile."""
    print(f"Generating graphs for {profile.benchmark_name} with {len(profile.batch_sizes)} batch sizes")
    
    # Convert profile data to BatchBenchResult objects
    metrics_data = []
    for i in range(len(profile.batch_sizes)):
        games_per_sec = profile.games_per_second[i] if profile.games_per_second else 0.0
        efficiency = profile.efficiency[i] if profile.efficiency else 0.0
        
        # Calculate missing values if needed
        avg_game_length = 1.0  # Default values for missing metrics
        median_game_length = 1.0
        min_game_length = 1.0
        max_game_length = 1.0
        
        result = BatchBenchResult(
            batch_size=profile.batch_sizes[i],
            steps_per_second=profile.steps_per_second[i],
            games_per_second=games_per_sec,
            avg_game_length=avg_game_length,
            median_game_length=median_game_length,
            min_game_length=min_game_length,
            max_game_length=max_game_length,
            memory_usage_gb=profile.memory_usage_gb[i],
            memory_usage_percent=profile.memory_usage_percent[i],
            efficiency=efficiency,
            valid=True
        )
        metrics_data.append(result)
    
    # Generate plots using existing function
    return generate_benchmark_plots(profile.benchmark_name, metrics_data, profile.timestamp)


def save_profile(profile: BenchmarkProfile, **kwargs) -> Path:
    """Save benchmark profile to a JSON file."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    filename = create_profile_filename(profile.benchmark_name, **kwargs)
    filepath = PROFILE_DIR / filename

    # Add current system information to the profile
    profile_dict = asdict(profile)
    sys_info = get_system_info()

    # Add all enhanced system info for better traceability
    enhanced_fields = {
        "device_name": sys_info["device_name"],
        "device_platform": sys_info["device_platform"],
        "device_kind": sys_info["device_kind"],
        "num_devices": sys_info["num_devices"],
        "cpu_count": sys_info["cpu_count"],
        "cpu_type": sys_info["cpu_type"],
        "pgx_version": sys_info["pgx_version"],
        "turbozero_version": sys_info["turbozero_version"],
        "turbozero_install_time": sys_info["turbozero_install_time"],
    }
    profile_dict.update(enhanced_fields)

    # Convert all values to JSON-serializable types (handles JAX arrays)
    profile_dict = convert_to_python(profile_dict)

    with open(filepath, 'w') as f:
        json.dump(profile_dict, f, indent=2)

    print(f"Profile saved to {filepath}")
    return filepath


def load_profile(benchmark_name: str = None, **kwargs) -> Optional[BenchmarkProfile]:
    """Load benchmark profile from a JSON file if it exists."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # List all profiles in the directory
    all_profiles = list(PROFILE_DIR.glob("*.json"))
    print(f"Found {len(all_profiles)} profile(s) in {PROFILE_DIR}:", flush=True)
    for profile_path in all_profiles:
        print(f"  - {profile_path.name}", flush=True)
    
    # If benchmark_name is provided, use the new matching logic
    if benchmark_name:
        filepath = find_matching_profile(benchmark_name, **kwargs)
        if not filepath:
            return None
    else:
        # Fallback to old logic for backward compatibility
        filename = create_profile_filename(**kwargs)
        filepath = PROFILE_DIR / filename
        
        if not filepath.exists():
            print(f"Warning: Could not find profile at {filepath}", flush=True)
            return None
    
    print(f"Loading profile from {filepath}", flush=True)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle missing duration field for backward compatibility
        if 'duration' not in data:
            data['duration'] = DEFAULT_BENCHMARK_DURATION
            print(f"Note: Adding default duration ({DEFAULT_BENCHMARK_DURATION}s) to loaded profile", flush=True)
        
        # Handle missing fields for backward compatibility
        if 'device_name' not in data:
            data['device_name'] = "unknown"
        if 'device_platform' not in data:
            data['device_platform'] = "unknown"
        if 'num_devices' not in data:
            data['num_devices'] = 1
        
        # Filter data to only include fields that BenchmarkProfile expects
        from dataclasses import fields
        profile_fields = {field.name for field in fields(BenchmarkProfile)}
        filtered_data = {k: v for k, v in data.items() if k in profile_fields}
        
        return BenchmarkProfile(**filtered_data)
    except Exception as e:
        print(f"Error loading profile: {e}", flush=True)
        return None


def get_cpu_gpu_usage() -> Tuple[float, float]:
    """Get CPU and GPU usage percentages."""
    cpu_usage = 0.0
    gpu_usage = 0.0
    
    # Get CPU usage
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)  # Sample over 0.1s
    except ImportError:
        print("Warning: psutil not installed, cannot measure CPU usage", flush=True)
    
    # Get GPU usage if available
    sys_info = get_system_info()
    if sys_info["jaxlib_type"] == "cuda":
        try:
            device = jax.devices()[0]
            device_id = device.id
            cmd = [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu",
                f"--format=csv,noheader,nounits",
                f"-i", f"{device_id}"
            ]
            output = subprocess.check_output(cmd, timeout=2).decode().strip()
            gpu_usage = float(output)
        except Exception as e:
            print(f"Warning: Failed to get GPU usage via nvidia-smi: {e}", flush=True)
    
    return cpu_usage, gpu_usage


def random_action_from_mask(key, mask):
    """Sample a random action index based on the legal action mask."""
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits)


def create_results_dir(name: str = "graphs") -> Path:
    """Create a directory for benchmark results if it doesn't exist."""
    results_dir = Path("benchmarks") / name
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def generate_benchmark_plots(
    benchmark_name: str,
    metrics_data: List[BatchBenchResult], 
    timestamp: Optional[str] = None
) -> Tuple[str, str]:
    """Generate standardized benchmark plots.
    
    Creates two plots:
    1. Performance plot: Batch Size vs Steps/Second with Games/Second on secondary axis
    2. Memory plot: Batch Size vs Memory Usage Percentage with Steps/Second on secondary axis
    
    Args:
        benchmark_name: Name of the benchmark for plot titles
        metrics_data: List of benchmark results
        timestamp: Optional string to include in filenames
    
    Returns:
        Tuple of (performance_plot_path, memory_plot_path)
    """
    results_dir = create_results_dir()
    
    system_info = get_system_info()
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower().split()[0]  # Take just the first part
    
    # Create a simple timestamp if none provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # Clean up timestamp to ensure valid filename
        timestamp = "".join(c for c in timestamp if c.isalnum() or c in "_-")[:50]  # Limit length
    
    # Extract metrics for plotting
    batch_sizes = [result.batch_size for result in metrics_data]
    steps_per_second = [result.steps_per_second for result in metrics_data]
    games_per_second = [result.games_per_second for result in metrics_data]
    memory_percent = [result.memory_usage_percent for result in metrics_data]
    
    # Create performance plot: Batch Size vs Steps/Second with Games/Second on secondary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis: Steps per second
    color1 = 'tab:blue'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Steps per Second', color=color1)
    ax1.plot(batch_sizes, steps_per_second, 'o-', color=color1, linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{benchmark_name} Performance: Batch Size vs Steps/Second')
    
    # Secondary axis: Games per second
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Games per Second', color=color2)
    ax2.plot(batch_sizes, games_per_second, '^-', color=color2, linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(['Steps/s'], loc='upper left')
    ax2.legend(['Games/s'], loc='upper right')
    
    plt.tight_layout()
    
    # Save performance plot
    base_filename = f"{platform_name}_{processor_name}_{benchmark_name.lower()}_{timestamp}"
    performance_plot_path = results_dir / f"{base_filename}_performance.png"
    plt.savefig(performance_plot_path, dpi=120)
    plt.close()
    
    # Create memory plot: Batch Size vs Memory Usage Percentage with Steps/Second on secondary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis: Memory usage percentage
    color1 = 'tab:green'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Memory Usage (%)', color=color1)
    ax1.plot(batch_sizes, memory_percent, 's-', color=color1, linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{benchmark_name} Memory Usage: Batch Size vs Memory Percentage')
    ax1.set_ylim(0, max(100, max(memory_percent) * 1.1))  # Set y-axis to show 0-100% or higher if needed
    
    # Secondary axis: Steps per second (for reference)
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Steps per Second', color=color2)
    # Normalize steps/second to fit nicely on the memory percentage scale
    max_steps = max(steps_per_second)
    normalized_steps = [s / max_steps * max(memory_percent) for s in steps_per_second]
    ax2.plot(batch_sizes, normalized_steps, 'o--', color=color2, alpha=0.7, linewidth=1, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    # Set secondary y-axis labels to show actual steps/second values
    ax2_ticks = ax2.get_yticks()
    ax2_labels = [f'{int(tick * max_steps / max(memory_percent)):,}' for tick in ax2_ticks]
    ax2.set_yticklabels(ax2_labels)
    
    # Add legend
    ax1.legend(['Memory %'], loc='upper left')
    ax2.legend(['Steps/s'], loc='upper right')
    
    plt.tight_layout()
    
    # Save memory plot
    memory_plot_path = results_dir / f"{base_filename}_memory.png"
    plt.savefig(memory_plot_path, dpi=120)
    plt.close()
    
    print(f"Performance plot saved to: {performance_plot_path}")
    print(f"Memory plot saved to: {memory_plot_path}")
    
    return str(performance_plot_path), str(memory_plot_path)


def validate_against_profile(
    batch_results: List[BatchBenchResult],
    profile_path: Path,
    system_info: Dict[str, str]
) -> None:
    """Compare current benchmark results against a previous profile."""
    if not profile_path.exists():
        print(f"No profile found at {profile_path}")
        return
    
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    if 'batch_sizes' not in profile or 'moves_per_second' not in profile:
        print("Profile doesn't contain required data for comparison")
        return
    
    # Extract current results
    current_batch_sizes = [result.batch_size for result in batch_results]
    current_moves_per_second = [result.moves_per_second for result in batch_results]
    
    # Create comparison directory if it doesn't exist
    results_dir = create_results_dir("graphs/comparisons")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract platform info for filenames
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower()
    
    # Find common batch sizes for comparison
    common_batch_sizes = []
    profile_moves = []
    current_moves = []
    
    for i, b in enumerate(current_batch_sizes):
        if b in profile['batch_sizes']:
            idx = profile['batch_sizes'].index(b)
            common_batch_sizes.append(b)
            profile_moves.append(profile['moves_per_second'][idx])
            current_moves.append(current_moves_per_second[i])
    
    if not common_batch_sizes:
        print("No common batch sizes found for comparison")
        return
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(common_batch_sizes))
    
    plt.bar(x - width/2, profile_moves, width, label='Previous')
    plt.bar(x + width/2, current_moves, width, label='Current')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Moves per Second')
    plt.title('Performance Comparison: Previous vs Current')
    plt.xticks(x, common_batch_sizes)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = results_dir / f"{platform_name}_{processor_name}_comparison_{timestamp}.png"
    plt.savefig(comparison_path, dpi=120)
    plt.close()
    
    # Create a difference plot
    percentage_diff = [(c - p) / p * 100 for p, c in zip(profile_moves, current_moves)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(common_batch_sizes, percentage_diff)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Performance Difference (%)')
    plt.title('Performance Difference: (Current - Previous) / Previous * 100%')
    
    diff_path = results_dir / f"{platform_name}_{processor_name}_diff_{timestamp}.png"
    plt.savefig(diff_path, dpi=120)
    plt.close()
    
    print(f"Comparison plot saved to: {comparison_path}")
    print(f"Difference plot saved to: {diff_path}")
    
    # Print summary
    avg_diff = sum(percentage_diff) / len(percentage_diff)
    print(f"\n=== Performance Comparison Summary ===")
    print(f"Average performance difference: {avg_diff:.2f}%")
    
    if avg_diff > 5:
        print("Performance IMPROVED compared to previous profile")
    elif avg_diff < -5:
        print("Performance DEGRADED compared to previous profile")
    else:
        print("Performance is similar to previous profile")


def print_benchmark_summary(results: List[BatchBenchResult]) -> None:
    """Print a formatted summary of benchmark results."""
    valid_results = [r for r in results if r.valid]
    
    if not valid_results:
        print("No valid benchmark results to display")
        return
    
    print("\n=== Discovery Summary (Valid Results) ===")
    header = f"{'Batch':>7} | {'Moves/s':>12} | {'Games/s':>12} | {'Moves/s/G':>10} | {'Mem (GB)':>10} | " \
             f"{'Effic.':>12} | {'Avg Moves':>10} | {'Med Moves':>10} | {'Min Moves':>10} | {'Max Moves':>10}"
    print(header)
    print("-" * 100)
    
    for result in valid_results:
        print(f"{result.batch_size:>7} | {result.moves_per_second:>12,.2f} | {result.games_per_second:>12,.2f} | "
              f"{result.moves_per_second / result.batch_size:>10,.2f} | {result.memory_usage_gb:>10,.2f} | "
              f"{result.efficiency:>12,.2f}/GB | {result.avg_game_length:>10,.1f} | {result.median_game_length:>10,.1f} | "
              f"{result.min_game_length:>10} | {result.max_game_length:>10}")
    
    # Print optimal configurations
    moves_optimal = max(valid_results, key=lambda x: x.moves_per_second)
    games_optimal = max(valid_results, key=lambda x: x.games_per_second)
    efficiency_optimal = max(valid_results, key=lambda x: x.efficiency)
    
    print("\n=== Optimal Configurations (Discovered) ===")
    print(f"Best for moves/s: Batch size {moves_optimal.batch_size} with {moves_optimal.moves_per_second:,.2f} moves/s")
    print(f"Best for games/s: Batch size {games_optimal.batch_size} with {games_optimal.games_per_second:,.2f} games/s")
    print(f"Best for efficiency: Batch size {efficiency_optimal.batch_size} with {efficiency_optimal.efficiency:,.2f}/GB")


def select_batch_sizes_for_profile(results: List[BatchBenchResult], num_sizes: int = 4) -> List[int]:
    """Select a meaningful subset of batch sizes for inclusion in the profile."""
    valid_results = [r for r in results if r.valid]
    
    if len(valid_results) <= num_sizes:
        return [r.batch_size for r in valid_results]
    
    # Always include the smallest and largest batch sizes
    smallest = min(valid_results, key=lambda x: x.batch_size).batch_size
    largest = max(valid_results, key=lambda x: x.batch_size).batch_size
    
    # Include the batch size with the best raw performance
    best_perf = max(valid_results, key=lambda x: x.moves_per_second).batch_size
    
    # Include the batch size with the best games/s if different
    best_games = max(valid_results, key=lambda x: x.games_per_second).batch_size
    
    # Start with these candidates
    selected = [smallest, largest, best_perf, best_games]
    selected = list(set(selected))  # Remove duplicates
    
    # If we need more, add intermediate sizes evenly distributed
    batch_sizes = sorted([r.batch_size for r in valid_results])
    
    if len(selected) < num_sizes:
        # How many more do we need?
        needed = num_sizes - len(selected)
        
        # Filter out already selected batch sizes
        remaining = [b for b in batch_sizes if b not in selected]
        
        if remaining:
            # Take evenly spaced items
            indices = np.linspace(0, len(remaining) - 1, needed).astype(int)
            additional = [remaining[i] for i in indices]
            selected.extend(additional)
    
    return sorted(selected[:num_sizes])  # Return sorted list, capped at num_sizes 


class BaseBenchmark:
    """Base class for all benchmarks providing common functionality."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.system_info = get_system_info()
    
    def warmup_compilation(self, step_fn, states, num_warmup=4):
        """Common warmup and compilation logic."""
        print("Compiling and warming up...", flush=True)
        key = jax.random.PRNGKey(0)
        
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            # Pass states as a tuple
            new_states = step_fn(subkey, states[0], states[1])
            jax.block_until_ready(new_states)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            for _ in range(num_warmup):
                key, subkey = jax.random.split(key)
                # Pass states as a tuple
                new_states = step_fn(subkey, new_states[0], new_states[1])
            jax.block_until_ready(new_states)
            print("Warm-up complete", flush=True)
            return new_states
            
        except Exception as e:
            print(f"Error during compilation/warm-up: {e}", flush=True)
            raise
    
    def run_benchmark_iteration(self, step_fn, states, pbar, start_time, max_duration):
        """Run a single benchmark iteration with progress tracking."""
        current_time = time.time()
        if current_time - start_time >= max_duration:
            return None
            
        try:
            # Execute step
            key = jax.random.PRNGKey(0)
            key, step_key = jax.random.split(key)
            new_states = step_fn(step_key, *states)
            jax.block_until_ready(new_states)
            
            # Update progress
            elapsed = current_time - start_time
            pbar.n = round(elapsed)
            pbar.refresh()
            
            return new_states
            
        except Exception as e:
            print(f"Error during benchmark iteration: {e}", flush=True)
            return None
    
    def save_profile(self, results: List[BatchBenchResult], extra_info: Dict[str, Any] = None):
        """Save benchmark results to a profile."""
        # Convert JAX arrays to Python types
        def convert_to_python(value):
            if isinstance(value, (jnp.ndarray, jax.Array)):
                return float(value)  # Convert to Python float
            if isinstance(value, (list, tuple)):
                return [convert_to_python(v) for v in value]
            if isinstance(value, dict):
                return {k: convert_to_python(v) for k, v in value.items()}
            return value

        profile_data = {
            # System info
            "platform": self.system_info["platform"],
            "processor": self.system_info["processor"],
            "jaxlib_type": self.system_info["jaxlib_type"],
            "device_info": self.system_info["device_info"],
            "python_version": self.system_info["python_version"],
            "jax_version": self.system_info["jax_version"],
            
            # Benchmark info
            "name": self.name,
            "description": self.description,
            "timestamp": datetime.now().isoformat(),
            
            # Results
            "batch_sizes": [r.batch_size for r in results],
            "moves_per_second": [float(r.moves_per_second) for r in results],
            "games_per_second": [float(r.games_per_second) for r in results],
            "memory_usage_gb": [float(r.memory_usage_gb) for r in results],
        }
        
        # Add any extra info
        if extra_info:
            profile_data.update({k: convert_to_python(v) for k, v in extra_info.items()})
        
        # Create filename
        filename = f"{self.system_info['platform'].lower()}_{self.name.lower()}.json"
        filepath = PROFILE_DIR / filename
        
        # Save
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"Profile saved to {filepath}")
        return filepath
    
    def load_profile(self, **kwargs) -> Optional[Dict[str, Any]]:
        """Load a matching profile for this benchmark."""
        filepath = find_matching_profile(self.name, **kwargs)
        
        if filepath:
            print(f"Found matching profile: {filepath}", flush=True)
            with open(filepath, 'r') as f:
                return json.load(f)
        
        print(f"No matching profile found for {self.name}", flush=True)
        return None
    
    def discover_and_save(self, 
                         memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
                         duration: int = DEFAULT_BENCHMARK_DURATION,
                         custom_batch_sizes: Optional[List[int]] = None,
                         force_overwrite: bool = False,
                         verbose: bool = False,
                         profile_params: Optional[Dict] = None) -> List[BatchBenchResult]:
        """Standardized discovery workflow for all benchmarks."""
        print(f"\n=== {self.name} Benchmark Discovery ===")
        print_system_info(self.system_info)
        
        # Check for existing profile
        kwargs = profile_params or {}
        profile = self.load_profile(**kwargs)
        
        # Determine if we need to force overwrite
        need_force = False
        if profile:
            if custom_batch_sizes:
                # Check if any of the custom batch sizes conflict with existing ones
                existing_batch_sizes = set(profile.get('batch_sizes', []))
                custom_batch_set = set(custom_batch_sizes)
                conflicts = existing_batch_sizes & custom_batch_set
                
                if conflicts:
                    need_force = True
                    if not force_overwrite:
                        print(f"Profile already exists for {self.name} with conflicting batch sizes: {sorted(conflicts)}")
                        print("Use --force to overwrite conflicting batch sizes or choose different batch sizes.")
                        return []
                    else:
                        print(f"Overwriting existing profile with conflicting batch sizes: {sorted(conflicts)}")
                else:
                    print(f"Adding new batch sizes {sorted(custom_batch_set - existing_batch_sizes)} to existing profile")
            else:
                if not force_overwrite:
                    print(f"Profile already exists for {self.name}. Use --force to overwrite.")
                    return []
        
        # Run discovery
        print(f"\nDiscovering optimal batch sizes (memory limit: {memory_limit_gb:.1f}GB, duration: {duration}s)")
        
        if custom_batch_sizes:
            print(f"Using custom batch sizes: {custom_batch_sizes}")
        else:
            print("Using progressive batch size discovery: 1, 2, 4, 8, 16, 32, ...")
        
        # This should be implemented by subclasses
        results = self._run_discovery(memory_limit_gb, duration, custom_batch_sizes, verbose)
        
        if results:
            # If we have an existing profile and we're adding batch sizes (not forcing), merge results
            if profile and custom_batch_sizes and not need_force:
                results = self._merge_with_existing_profile(results, profile)
            
            # Print summary
            self.print_results_summary(results)
            
            # Generate plots
            perf_plot, mem_plot = generate_benchmark_plots(self.name, results)
            
            # Save profile
            self._save_standardized_profile(results, profile_params)
            
            print(f"\n✓ Discovery complete for {self.name}")
            print(f"  - {len(results)} batch sizes tested")
            print(f"  - Plots saved: {Path(perf_plot).name}, {Path(mem_plot).name}")
        
        return results
    
    def _merge_with_existing_profile(self, new_results: List[BatchBenchResult], profile: Dict[str, Any]) -> List[BatchBenchResult]:
        """Merge new results with existing profile data."""
        print(f"Merging new results with existing profile...")
        
        # Extract existing data
        existing_batch_sizes = profile.get('batch_sizes', [])
        existing_steps = profile.get('steps_per_second', [])
        existing_games = profile.get('games_per_second', [])
        existing_memory_gb = profile.get('memory_usage_gb', [])
        existing_memory_percent = profile.get('memory_usage_percent', [])
        existing_efficiency = profile.get('efficiency', [])
        
        # Create result objects from existing data
        existing_results = []
        for i, batch_size in enumerate(existing_batch_sizes):
            # Skip if this batch size is in new results (we'll use the new data)
            if any(r.batch_size == batch_size for r in new_results):
                continue
            
            # Create a result object from existing data
            result = BatchBenchResult(
                batch_size=batch_size,
                steps_per_second=existing_steps[i] if i < len(existing_steps) else 0.0,
                games_per_second=existing_games[i] if i < len(existing_games) else 0.0,
                avg_game_length=0.0,  # Not stored in old profiles
                median_game_length=0.0,
                min_game_length=0,
                max_game_length=0,
                memory_usage_gb=existing_memory_gb[i] if i < len(existing_memory_gb) else 0.0,
                memory_usage_percent=existing_memory_percent[i] if i < len(existing_memory_percent) else 0.0,
                efficiency=existing_efficiency[i] if i < len(existing_efficiency) else 0.0,
                valid=True
            )
            existing_results.append(result)
        
        # Combine and sort by batch size
        all_results = existing_results + new_results
        all_results.sort(key=lambda r: r.batch_size)
        
        print(f"Merged profile now contains {len(all_results)} batch sizes")
        return all_results
    
    def validate_performance(self,
                           memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
                           duration: int = DEFAULT_BENCHMARK_DURATION,
                           verbose: bool = False,
                           profile_params: Optional[Dict] = None) -> bool:
        """Standardized validation workflow for all benchmarks."""
        print(f"\n=== {self.name} Benchmark Validation ===")
        print_system_info(self.system_info)
        
        # Load existing profile
        kwargs = profile_params or {}
        profile = self.load_profile(**kwargs)
        if not profile:
            print(f"No existing profile found for {self.name}. Run discovery first.")
            return False
        
        print(f"Validating against profile from {profile.get('timestamp', 'unknown date')}")
        
        # Run validation with same batch sizes
        batch_sizes = profile.get('batch_sizes', [])
        if not batch_sizes:
            print("No batch sizes found in profile")
            return False
        
        print(f"Testing batch sizes: {batch_sizes}")
        results = self._run_discovery(memory_limit_gb, duration, batch_sizes, verbose)
        
        if results:
            # Compare with existing profile
            self._compare_with_profile(results, profile)
            return True
        
        return False
    
    def print_single_result(self, result: BatchBenchResult, extra_info: Optional[Dict[str, Any]] = None):
        """Print a single benchmark result."""
        print(f"\n=== {self.name} Single Batch Result ===")
        print(f"Batch Size: {result.batch_size}")
        print(f"Steps/Second: {result.steps_per_second:,.2f}")
        print(f"Games/Second: {result.games_per_second:,.2f}")
        print(f"Memory Usage: {result.memory_usage_gb:.2f} GB ({result.memory_usage_percent:.1f}%)")
        print(f"Efficiency: {result.efficiency:.2f} steps/s/GB")
        if extra_info:
            for key, value in extra_info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    def print_results_summary(self, results: List[BatchBenchResult]):
        """Print a formatted summary of all results."""
        if not results:
            print("No results to display")
            return
        
        print(f"\n=== {self.name} Discovery Summary ===")
        header = (
            f"{'Batch':>7} | {'Steps/s':>12} | {'Games/s':>12} | {'Mem %':>8} | "
            f"{'Efficiency':>12} | {'Avg Length':>11}"
        )
        print(header)
        print("-" * len(header))
        
        for result in results:
            print(
                f"{result.batch_size:>7} | "
                f"{format_human_readable(result.steps_per_second):>12} | "
                f"{format_human_readable(result.games_per_second):>12} | "
                f"{result.memory_usage_percent:>7.1f}% | "
                f"{result.efficiency:>12.2f} | "
                f"{result.avg_game_length:>11.1f}"
            )
        
        # Print optimal configurations
        best_steps = max(results, key=lambda x: x.steps_per_second)
        best_efficiency = max(results, key=lambda x: x.efficiency)
        
        print(f"\n=== Optimal Configurations ===")
        print(f"Best Steps/s: Batch {best_steps.batch_size} → {format_human_readable(best_steps.steps_per_second)}/s")
        print(f"Best Efficiency: Batch {best_efficiency.batch_size} → {best_efficiency.efficiency:.2f} steps/s/GB")
    
    def _run_discovery(self, 
                      memory_limit_gb: float, 
                      duration: int, 
                      custom_batch_sizes: Optional[List[int]],
                      verbose: bool) -> List[BatchBenchResult]:
        """Run the actual discovery - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_discovery")
    
    def _save_standardized_profile(self, results: List[BatchBenchResult], profile_params: Optional[Dict] = None):
        """Save results in standardized profile format."""
        profile_data = {
            # System info
            "platform": self.system_info["platform"],
            "processor": self.system_info["processor"],
            "jaxlib_type": self.system_info["jaxlib_type"],
            "device_info": self.system_info["device_info"],
            "python_version": self.system_info["python_version"],
            "jax_version": self.system_info["jax_version"],
            
            # Enhanced hardware identification
            "device_name": self.system_info["device_name"],
            "device_platform": self.system_info["device_platform"],
            "num_devices": self.system_info["num_devices"],
            
            # Benchmark info
            "benchmark_name": self.name,
            "description": self.description,
            "timestamp": datetime.now().isoformat(),
            
            # Results in standardized format
            "batch_sizes": [r.batch_size for r in results],
            "steps_per_second": [float(r.steps_per_second) for r in results],
            "games_per_second": [float(r.games_per_second) for r in results],
            "memory_usage_gb": [float(r.memory_usage_gb) for r in results],
            "memory_usage_percent": [float(r.memory_usage_percent) for r in results],
            "efficiency": [float(r.efficiency) for r in results],
        }
        
        # Create filename using new system
        kwargs = profile_params or {}
        filename = create_profile_filename(self.name, **kwargs)
        filepath = PROFILE_DIR / filename
        
        # Save
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"Profile saved to {filepath}")
        return filepath
    
    def _compare_with_profile(self, current_results: List[BatchBenchResult], profile: Dict[str, Any]):
        """Compare current results with existing profile."""
        print(f"\n=== Performance Comparison ===")
        
        # Extract profile data
        profile_batch_sizes = profile.get('batch_sizes', [])
        profile_steps = profile.get('steps_per_second', profile.get('moves_per_second', []))
        
        if not profile_steps:
            print("No performance data in profile for comparison")
            return
        
        # Find common batch sizes
        current_dict = {r.batch_size: r.steps_per_second for r in current_results}
        profile_dict = {bs: steps for bs, steps in zip(profile_batch_sizes, profile_steps)}
        
        common_sizes = set(current_dict.keys()) & set(profile_dict.keys())
        
        if not common_sizes:
            print("No common batch sizes found for comparison")
            return
        
        print(f"{'Batch':>7} | {'Previous':>12} | {'Current':>12} | {'Change':>12}")
        print("-" * 50)
        
        total_change = 0
        count = 0
        
        for batch_size in sorted(common_sizes):
            prev = profile_dict[batch_size]
            curr = current_dict[batch_size]
            change = ((curr - prev) / prev) * 100
            total_change += change
            count += 1
            
            change_str = f"{change:+.1f}%" if abs(change) < 999 else f"{change:+.0f}%"
            print(f"{batch_size:>7} | {format_human_readable(prev):>12} | "
                  f"{format_human_readable(curr):>12} | {change_str:>12}")
        
        if count > 0:
            avg_change = total_change / count
            print(f"\nAverage performance change: {avg_change:+.1f}%")
            
            if avg_change > 5:
                print("✓ Performance IMPROVED compared to previous profile")
            elif avg_change < -5:
                print("⚠ Performance DEGRADED compared to previous profile")
            else:
                print("→ Performance is similar to previous profile") 

def print_summary_table(results: List[BatchBenchResult], title: str = None) -> None:
    """Print a formatted summary table of benchmark results."""
    if not results:
        print("No results to display")
        return
        
    if title:
        print(f"\n=== {title} ===")
    
    header = (
        f"{'Batch':>7} | {'Moves/s':>12} | {'Games/s':>12} | {'Mem (GB)':>10} | "
        f"{'Efficiency':>12} | {'Avg Moves':>10}"
    )
    print(header)
    print("-" * len(header))
    
    for result in results:
        print(
            f"{result.batch_size:>7} | "
            f"{format_human_readable(result.moves_per_second):>12} | "
            f"{format_human_readable(result.games_per_second):>12} | "
            f"{result.memory_usage_gb:>10.2f} | "
            f"{result.efficiency:>12.2f} | "
            f"{result.avg_game_length:>10.1f}"
        ) 