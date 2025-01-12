from backgammon_env import BackgammonEnv
from typing import Dict, Any
import time
import numpy as np
import gc
import multiprocessing as mp
import queue
import psutil

def player_name(player_num: int) -> str:
    return {
        0: "None",
        1: "White",
        2: "Red"
    }.get(player_num, "Unknown")

def print_move_info(move: Dict[str, Any]) -> None:
    print(f"Roll: {move['roll']} roll_used: {move['roll_used']} Player: {move['player']}")
    print(f"  White: {move['white_pips']}")
    print(f"    Red: {move['red_pips']}\n")

def test_basic_gameplay():
    print("Testing basic gameplay...")
    env = BackgammonEnv()
    state = env.get_encoded_state()
    print("Initial encoded state:", state)
    
    moves = env.get_legal_moves()
    print(f"\nFound {len(moves)} legal moves")
    print("First move details:")
    if moves:
        print_move_info(moves[0])

    # Test move application
    if moves:
        next_state, victor, score = env.step(0)
        print(f"After move - Victor: {player_name(victor)}, Score: {score}")

def run_parallel_game(process_id: int, max_runtime: int, result_queue: mp.Queue):
    """Run games for a specific process and put results in queue.
    
    Args:
        process_id: ID of the current process
        max_runtime: Maximum runtime in seconds
        result_queue: Queue to store results
    """
    stats = {
        "total_games": 0,
        "normal_wins": 0,
        "gammons": 0,
        "backgammons": 0,
        "white_wins": 0,
        "red_wins": 0
    }
    
    env = BackgammonEnv(seed=42 + process_id)  # Different seed for each process
    env.set_white_ai(env.SMART_AI)  # White plays smart
    env.set_red_ai(env.SMART_AI)   # Red plays worst

    start_time = time.time()
    
    while time.time() - start_time < max_runtime:
        env.reset()
        moves_count = 0
        
        while moves_count < 200:  # Prevent infinite games
            state, victor, score = env.step_ai()
            moves_count += 1
            
            if victor:
                stats["total_games"] += 1
                if victor == 1:  # White wins
                    stats["white_wins"] += 1
                    if score == 1:
                        stats["normal_wins"] += 1
                    elif score == 2:
                        stats["gammons"] += 1
                    elif score == 3:
                        stats["backgammons"] += 1
                        # Print immediately when backgammon is found
                        #print(f"\nProcess {process_id}: Backgammon found!")
                elif victor == 2:  # Red wins
                    stats["red_wins"] += 1
                break
                
        # Update progress every 100 games
        if stats["total_games"] % 100 == 0:
            result_queue.put(("progress", process_id, stats.copy()))
    
    # Send final results
    result_queue.put(("final", process_id, stats))

def test_find_special_wins(max_runtime: int = 30):
    """Test to find gammon and backgammon situations using parallel processing.
    
    Args:
        max_runtime: Maximum runtime in seconds per process
    """
    print("\nTesting for special win conditions using parallel processing...")
    
    # Initialize multiprocessing
    num_cores = mp.cpu_count()
    print(f"Running on {num_cores} CPU cores")
    
    # Create a queue for results
    result_queue = mp.Queue()
    
    # Start processes
    processes = []
    for i in range(num_cores):
        p = mp.Process(target=run_parallel_game, args=(i, max_runtime, result_queue))
        processes.append(p)
        p.start()
    
    # Initialize combined stats
    combined_stats = {
        "total_games": 0,
        "normal_wins": 0,
        "gammons": 0,
        "backgammons": 0,
        "white_wins": 0,
        "red_wins": 0
    }
    
    # Process stats dictionary for progress tracking
    process_stats = {i: combined_stats.copy() for i in range(num_cores)}
    
    start_time = time.time()
    active_processes = num_cores
    
    def print_progress(force=False):
        current_time = time.time()
        elapsed = current_time - start_time
        total_games = sum(ps["total_games"] for ps in process_stats.values())
        
        if not force and total_games % 100 != 0:
            return
            
        games_per_sec = total_games / elapsed if elapsed > 0 else 0
        progress = int((elapsed / max_runtime) * 20)
        bar = f"[{'=' * progress}{' ' * (20-progress)}]"
        
        total_stats = {k: sum(ps[k] for ps in process_stats.values()) 
                      for k in combined_stats.keys()}
        
        print(f"\r{bar} Games: {total_games:,} | "
              f"Rate: {games_per_sec:.1f}/s | "
              f"Time: {int(elapsed)}s/{max_runtime}s | "
              f"Wins: {total_stats['normal_wins']} | "
              f"Gammons: {total_stats['gammons']} | "
              f"Backgammons: {total_stats['backgammons']}", 
              end="", flush=True)
        return total_stats
    
    # Monitor results while processes are running
    while active_processes > 0:
        try:
            msg_type, process_id, stats = result_queue.get(timeout=0.1)
            if msg_type == "final":
                active_processes -= 1
            process_stats[process_id] = stats
            print_progress()
        except queue.Empty:
            continue
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Get final combined statistics
    final_stats = print_progress(force=True)
    
    # Print final statistics
    print("\n\n" + "="*50)
    print("Final Statistics (All Cores):")
    print("="*50)
    total_time = time.time() - start_time
    print(f"Runtime:         {total_time:.1f}s")
    print(f"Total Games:     {final_stats['total_games']:,}")
    print(f"Games/Second:    {final_stats['total_games'] / total_time:.1f}")
    print("\nWin Distribution:")
    print(f"White Wins:      {final_stats['white_wins']:,}")
    print(f"Red Wins:        {final_stats['red_wins']:,}")
    print("\nWin Types:")
    print(f"Normal Wins:     {final_stats['normal_wins']:,}")
    print(f"Gammons:         {final_stats['gammons']:,}")
    print(f"Backgammons:     {final_stats['backgammons']:,}")
    print("="*50)

def test_ai_gameplay():
    print("\nTesting AI gameplay...")
    env = BackgammonEnv()
    moves_count = 0
    start_time = time.time()
    
    
    while True:
        state, victor, score = env.step_ai()
        moves_count += 1
        
        if victor:
            elapsed = time.time() - start_time
            print(f"Game over after {moves_count} moves ({elapsed:.2f}s)")
            print(f"Victor: {player_name(victor)}, Score: {score}")
            break
        
        if moves_count >= 200:  # Safety limit
            print("Game exceeded move limit")
            break

def test_self_play():
    print("\nTesting self-play functionality...")
    env = BackgammonEnv()
    game_stats = {
        "moves": 0,
        "ai_moves": 0,
        "player_moves": 0,
        "total_time": 0
    }
    
    start_time = time.time()
    
    # Alternate between AI and player moves
    while True:
        # AI move
        state, victor, score = env.step_ai()
        game_stats["ai_moves"] += 1
        game_stats["moves"] += 1
        
        if victor:
            break
            
        # Get legal moves for player
        moves = env.get_legal_moves()
        if not moves:
            break
            
        # Choose first legal move for player
        state, victor, score = env.step(0)
        game_stats["player_moves"] += 1
        game_stats["moves"] += 1
        
        if victor:
            break
            
        if game_stats["moves"] >= 200:  # Safety limit
            print("Game exceeded move limit")
            break
    
    game_stats["total_time"] = time.time() - start_time
    print(f"Self-play game completed:")
    print(f"Total moves: {game_stats['moves']}")
    print(f"AI moves: {game_stats['ai_moves']}")
    print(f"Player moves: {game_stats['player_moves']}")
    print(f"Time taken: {game_stats['total_time']:.2f}s")
    print(f"Victor: {player_name(victor)}, Score: {score}")

def test_move_caching():
    print("\nTesting move caching...")
    env = BackgammonEnv()
    
    # Get legal moves first time
    start_time = time.time()
    moves1 = env.get_legal_moves()
    time1 = time.time() - start_time
    
    # Get legal moves second time (should be cached)
    start_time = time.time()
    moves2 = env.get_legal_moves()
    time2 = time.time() - start_time
    
    print(f"First call time: {time1:.6f}s")
    print(f"Second call time (cached): {time2:.6f}s")
    print(f"Number of moves: {len(moves1)}")
    assert moves1 == moves2, "Cached moves should match original moves"
    
    # Make a move and verify cache is invalidated
    if moves1:
        env.step(0)
        moves3 = env.get_legal_moves()
        assert moves3 != moves1, "Moves should be different after board change"
        print("Cache successfully invalidated after move")

def test_memory_usage():
    """Test memory handling with actual memory measurements."""
    print("\nTesting memory handling...")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
    envs = []
    memory_samples = []
    
    # Create multiple environments and track memory
    for i in range(100):  # Increased to 100 environments for better measurement
        env = BackgammonEnv()
        moves = env.get_legal_moves()  # Force some memory allocation
        envs.append(env)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(current_memory)
        
        if i % 10 == 0:  # Print every 10th environment
            print(f"Created environment {i+1}, Memory usage: {current_memory:.2f}MB")
    
    max_memory = max(memory_samples)
    avg_memory = sum(memory_samples) / len(memory_samples)
    
    # Clean up environments
    for env in envs:
        del env
    
    # Force garbage collection
    import gc
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_diff = final_memory - initial_memory
    
    print("\nMemory Usage Statistics:")
    print(f"Initial Memory: {initial_memory:.2f}MB")
    print(f"Peak Memory: {max_memory:.2f}MB")
    print(f"Average Memory: {avg_memory:.2f}MB")
    print(f"Final Memory: {final_memory:.2f}MB")
    print(f"Memory Difference: {memory_diff:.2f}MB")
    
    # Check for memory leaks
    if memory_diff > 5:  # Alert if more than 5MB wasn't freed
        print("\nWARNING: Possible memory leak detected!")
    else:
        print("\nNo significant memory leaks detected.")

def main():
    print("Starting Backgammon Environment Tests\n")
    
    try:
        test_basic_gameplay()
        test_ai_gameplay()
        test_self_play()
        test_move_caching()
        test_memory_usage()
        test_find_special_wins()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()