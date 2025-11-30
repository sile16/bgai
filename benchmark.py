#!/usr/bin/env python3
"""
Unified benchmark CLI entry point for backgammon AI components.

This script provides a consistent interface for running benchmarks across
different components (game environment, MCTS, StochasticMCTS) with standardized
discovery and validation workflows.
"""

import sys
import argparse
from typing import Optional, List
from pathlib import Path

def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands for each benchmark type."""
    parser = argparse.ArgumentParser(
        description="Backgammon AI Benchmark Suite - Discover optimal batch sizes for efficient training and gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover optimal batch sizes for MCTS
  python benchmark.py mcts --discover

  # Validate MCTS performance against existing profile
  python benchmark.py mcts --validate

  # Test specific batch size for StochasticMCTS
  python benchmark.py stochastic-mcts --single-batch 16 --duration 60

  # Custom memory limit and batch sizes for game environment
  python benchmark.py game-env --discover --memory-limit 32 --batch-sizes "1,4,8,16,32"

  # Discover optimal batch sizes for two-player evaluation
  python benchmark.py two-player --discover

  # Benchmark training loop with custom parameters
  python benchmark.py training-loop --discover --train-batch-size 256 --collection-steps 5

  # Force overwrite existing profile
  python benchmark.py mcts --discover --force
        """
    )
    
    subparsers = parser.add_subparsers(dest='benchmark_type', help='Benchmark type to run')
    subparsers.required = True
    
    # Common arguments for all benchmarks
    def add_common_args(subparser):
        """Add common arguments to a subparser."""
        mode_group = subparser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument('--discover', action='store_true',
                              help='Discover optimal batch sizes through progressive testing')
        mode_group.add_argument('--validate', action='store_true', 
                              help='Validate performance against existing profile')
        mode_group.add_argument('--single-batch', type=int, metavar='SIZE',
                              help='Test only a specific batch size')
        
        subparser.add_argument('--memory-limit', type=float, default=24.0, metavar='GB',
                             help='Memory limit in GB (default: 24.0)')
        subparser.add_argument('--duration', type=int, default=30, metavar='SEC',
                             help='Duration of each batch size test in seconds (default: 30)')
        subparser.add_argument('--batch-sizes', type=str, metavar='SIZES',
                             help='Comma-separated list of batch sizes to test (e.g., "1,2,4,8,16")')
        subparser.add_argument('--force', action='store_true',
                             help='Force overwrite of existing benchmark profile')
        subparser.add_argument('--verbose', action='store_true',
                             help='Enable verbose output')
    
    # Game Environment benchmark
    game_env_parser = subparsers.add_parser('game-env', 
                                           help='Benchmark game environment step performance')
    add_common_args(game_env_parser)
    
    # MCTS benchmark  
    mcts_parser = subparsers.add_parser('mcts',
                                       help='Benchmark standard MCTS evaluator')
    add_common_args(mcts_parser)
    mcts_parser.add_argument('--num-simulations', type=int, default=200, metavar='N',
                           help='Number of MCTS simulations per move (default: 200)')
    mcts_parser.add_argument('--max-nodes', type=int, default=1200, metavar='N',
                           help='Maximum number of nodes in MCTS tree (default: 1200)')
    mcts_parser.add_argument('--iterations-list', type=str, metavar='LIST',
                           help='Comma-separated list of iteration counts (e.g., "200,600,1000")')
    mcts_parser.add_argument('--max-nodes-list', type=str, metavar='LIST',
                           help='Comma-separated list of max node counts (e.g., "1200,2400")')
    
    # StochasticMCTS benchmark
    stochastic_parser = subparsers.add_parser('stochastic-mcts',
                                            help='Benchmark stochastic MCTS evaluator')
    add_common_args(stochastic_parser)
    stochastic_parser.add_argument('--num-simulations', type=int, default=200, metavar='N',
                                 help='Number of MCTS simulations per move (default: 200)')
    stochastic_parser.add_argument('--max-nodes', type=int, default=1200, metavar='N',
                           help='Maximum number of nodes in MCTS tree (default: 1200)')
    stochastic_parser.add_argument('--iterations-list', type=str, metavar='LIST',
                           help='Comma-separated list of iteration counts (e.g., "200,600,1000")')
    stochastic_parser.add_argument('--max-nodes-list', type=str, metavar='LIST',
                           help='Comma-separated list of max node counts (e.g., "1200,2400")')
    
    # Two-Player Baseline benchmark
    two_player_parser = subparsers.add_parser('two-player',
                                            help='Benchmark two-player baseline performance')
    add_common_args(two_player_parser)
    
    # Training Loop benchmark
    training_parser = subparsers.add_parser('training-loop',
                                          help='Benchmark training loop performance')
    add_common_args(training_parser)
    training_parser.add_argument('--train-batch-size', type=int, default=512, metavar='SIZE',
                               help='Training batch size (default: 512)')
    training_parser.add_argument('--collection-steps', type=int, default=10, metavar='N',
                               help='Collection steps per benchmark iteration (default: 10)')
    training_parser.add_argument('--train-steps', type=int, default=5, metavar='N',
                               help='Training steps per benchmark iteration (default: 5)')
    
    # Simple Two-Player benchmark
    simple_two_player_parser = subparsers.add_parser('simple-two-player',
                                                    help='Benchmark simple two-player heuristic performance')
    add_common_args(simple_two_player_parser)
    
    # Model Comparison benchmark
    model_comp_parser = subparsers.add_parser('model-comparison',
                                             help='Benchmark model vs model head-to-head comparison')
    add_common_args(model_comp_parser)
    model_comp_parser.add_argument('--model1', type=str, default='neural_net', 
                                  choices=['neural_net', 'resnet_10layer', 'heuristic'],
                                  help='First model to compare (default: neural_net)')
    model_comp_parser.add_argument('--model2', type=str, default='heuristic',
                                  choices=['neural_net', 'resnet_10layer', 'heuristic'], 
                                  help='Second model to compare (default: heuristic)')
    model_comp_parser.add_argument('--num-simulations', type=int, default=300, metavar='N',
                                  help='Number of MCTS simulations per move (default: 300)')
    model_comp_parser.add_argument('--max-nodes', type=int, default=1000, metavar='N',
                                  help='Maximum number of nodes in MCTS tree (default: 1000)')
    model_comp_parser.add_argument('--model1-params', type=str, metavar='PATH',
                                  help='Path to checkpoint file for model 1 parameters')
    model_comp_parser.add_argument('--model2-params', type=str, metavar='PATH',
                                  help='Path to checkpoint file for model 2 parameters')
    
    return parser

def run_game_env_benchmark(args):
    """Run game environment benchmark."""
    from benchmarks.bench_game_env import GameEnvironmentBenchmark
    
    benchmark = GameEnvironmentBenchmark()
    
    if args.single_batch:
        result = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result)
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    elif args.discover:
        results = benchmark.discover_and_save(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
            force_overwrite=args.force,
            verbose=args.verbose
        )
    elif args.validate:
        benchmark.validate_performance(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            verbose=args.verbose
        )

def run_mcts_benchmark(args):
    """Run MCTS benchmark."""
    from benchmarks.bench_mcts import MCTSBenchmark
    
    if args.single_batch:
        # For single batch, use the specified configuration
        benchmark = MCTSBenchmark(args.num_simulations, args.max_nodes)
        profile_params = benchmark.get_profile_params()
        
        result, nodes = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result, extra_info={'max_node_count': nodes})
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name, **profile_params)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name, **profile_params)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    else:
        # For discover/validate, run all configurations
        configs = get_mcts_configurations(args)
        print(f"Running MCTS benchmark with {len(configs)} configurations: {configs}")
        
        for i, (num_simulations, max_nodes) in enumerate(configs):
            print(f"\n=== Configuration {i+1}/{len(configs)}: {num_simulations} iterations, {max_nodes} max nodes ===")
            
            benchmark = MCTSBenchmark(num_simulations, max_nodes)
            profile_params = benchmark.get_profile_params()
            
            if args.discover:
                results, max_nodes_used = benchmark.discover_and_save(
                    memory_limit_gb=args.memory_limit,
                    duration=args.duration,
                    custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
                    force_overwrite=args.force,
                    verbose=args.verbose,
                    profile_params=profile_params
                )
            elif args.validate:
                benchmark.validate_performance(
                    memory_limit_gb=args.memory_limit,
                    duration=args.duration,
                    verbose=args.verbose,
                    profile_params=profile_params
                )

def run_stochastic_mcts_benchmark(args):
    """Run StochasticMCTS benchmark."""
    from benchmarks.bench_stochastic_mcts import StochasticMCTSBenchmark
    
    if args.single_batch:
        # For single batch, use the specified configuration
        benchmark = StochasticMCTSBenchmark(args.num_simulations, args.max_nodes)
        profile_params = benchmark.get_profile_params()
        
        result, nodes = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result, extra_info={'max_node_count': nodes})
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name, **profile_params)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name, **profile_params)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    else:
        # For discover/validate, run all configurations
        configs = get_mcts_configurations(args)
        print(f"Running StochasticMCTS benchmark with {len(configs)} configurations: {configs}")
        
        for i, (num_simulations, max_nodes) in enumerate(configs):
            print(f"\n=== Configuration {i+1}/{len(configs)}: {num_simulations} iterations, {max_nodes} max nodes ===")
            
            benchmark = StochasticMCTSBenchmark(num_simulations, max_nodes)
            profile_params = benchmark.get_profile_params()
            
            if args.discover:
                results, max_nodes_used = benchmark.discover_and_save(
                    memory_limit_gb=args.memory_limit,
                    duration=args.duration,
                    custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
                    force_overwrite=args.force,
                    verbose=args.verbose,
                    profile_params=profile_params
                )
            elif args.validate:
                benchmark.validate_performance(
                    memory_limit_gb=args.memory_limit,
                    duration=args.duration,
                    verbose=args.verbose,
                    profile_params=profile_params
                )

def run_two_player_benchmark(args):
    """Run two-player baseline benchmark."""
    from benchmarks.bench_two_player import TwoPlayerBaselineBenchmark
    
    benchmark = TwoPlayerBaselineBenchmark()
    
    if args.single_batch:
        result = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result)
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    elif args.discover:
        results = benchmark.discover_and_save(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
            force_overwrite=args.force,
            verbose=args.verbose
        )
    elif args.validate:
        benchmark.validate_performance(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            verbose=args.verbose
        )

def run_training_loop_benchmark(args):
    """Run training loop benchmark."""
    from benchmarks.bench_training_loop import TrainingLoopBenchmark
    
    benchmark = TrainingLoopBenchmark(
        train_batch_size=args.train_batch_size,
        collection_steps=args.collection_steps,
        train_steps=args.train_steps
    )
    profile_params = benchmark.get_profile_params()
    
    if args.single_batch:
        result = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result)
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name, **profile_params)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name, **profile_params)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    elif args.discover:
        results = benchmark.discover_and_save(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
            force_overwrite=args.force,
            verbose=args.verbose,
            profile_params=profile_params
        )
    elif args.validate:
        benchmark.validate_performance(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            verbose=args.verbose,
            profile_params=profile_params
        )

def run_simple_two_player_benchmark(args):
    """Run simple two-player benchmark."""
    from benchmarks.bench_simple_two_player import SimpleTwoPlayerBenchmark
    
    benchmark = SimpleTwoPlayerBenchmark()
    
    if args.single_batch:
        result = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result)
        
        # Insert result into existing profile and generate graphs
        from benchmarks.benchmark_common import insert_batch_result_to_profile, load_profile, generate_graphs_from_profile
        
        updated_profile_path = insert_batch_result_to_profile(result, benchmark.name)
        if updated_profile_path:
            print(f"\nUpdated profile: {updated_profile_path}")
            
            # Load the updated profile and generate graphs
            updated_profile = load_profile(benchmark.name)
            if updated_profile:
                print("Generating graphs from complete profile data...")
                try:
                    perf_plot, mem_plot = generate_graphs_from_profile(updated_profile)
                    print(f"Performance plot: {perf_plot}")
                    print(f"Memory plot: {mem_plot}")
                except Exception as e:
                    print(f"Error generating graphs: {e}")
            else:
                print("Could not load updated profile for graph generation")
        else:
            print("No existing profile found - run discovery mode first to create a baseline profile")
    elif args.discover:
        results = benchmark.discover_and_save(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
            force_overwrite=args.force,
            verbose=args.verbose
        )
    elif args.validate:
        benchmark.validate_performance(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            verbose=args.verbose
        )

def run_model_comparison_benchmark(args):
    """Run model comparison benchmark."""
    from benchmarks.bench_model_comparison import ModelComparisonBenchmark
    
    benchmark = ModelComparisonBenchmark(
        model1_name=args.model1,
        model2_name=args.model2,
        num_simulations=args.num_simulations,
        max_nodes=args.max_nodes,
        model1_params_path=getattr(args, 'model1_params', None),
        model2_params_path=getattr(args, 'model2_params', None)
    )
    
    if args.single_batch:
        result = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        benchmark.print_single_result(result)
    elif args.discover:
        results = benchmark._run_discovery(
            memory_limit_gb=args.memory_limit,
            duration=args.duration,
            custom_batch_sizes=parse_batch_sizes(args.batch_sizes),
            verbose=args.verbose
        )
        
        # Print summary
        print(f"\n=== Model Comparison Discovery Complete ===")
        print(f"Tested {len(results)} batch sizes")
        if results:
            best_perf = max(results, key=lambda x: x['games_per_second'])
            print(f"Best performance: {best_perf['games_per_second']:.2f} games/s at batch size {best_perf['batch_size']}")
            print(f"Final reward difference ({args.model1} - {args.model2}): {best_perf['reward_difference']:.3f}")
    elif args.validate:
        print("Validation mode not implemented for model comparison")
        return

def parse_batch_sizes(batch_sizes_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated batch sizes string into a list of integers."""
    if not batch_sizes_str:
        return None
        
    try:
        return [int(b.strip()) for b in batch_sizes_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Error parsing batch sizes: {e}. Format should be comma-separated integers (e.g., '1,2,4,8,16').")

def parse_int_list(list_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integer list."""
    if not list_str:
        return None
        
    try:
        return [int(x.strip()) for x in list_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Error parsing integer list: {e}. Format should be comma-separated integers.")

def get_mcts_configurations(args):
    """Get MCTS configuration combinations from command line arguments."""
    # Get iterations list
    if args.iterations_list:
        iterations_list = parse_int_list(args.iterations_list)
    else:
        iterations_list = [200, 600, 1000]  # Default configurations
    
    # Get max_nodes list
    if args.max_nodes_list:
        max_nodes_list = parse_int_list(args.max_nodes_list)
    else:
        max_nodes_list = [1200, 2400]  # Default configurations
    
    # Create all combinations
    configs = []
    for iterations in iterations_list:
        for max_nodes in max_nodes_list:
            configs.append((iterations, max_nodes))
    
    return configs

def main():
    """Main entry point for the unified benchmark CLI."""
    parser = create_main_parser()
    args = parser.parse_args()
    
    try:
        # Route to appropriate benchmark based on type
        if args.benchmark_type == 'game-env':
            run_game_env_benchmark(args)
        elif args.benchmark_type == 'mcts':
            run_mcts_benchmark(args)
        elif args.benchmark_type == 'stochastic-mcts':
            run_stochastic_mcts_benchmark(args)
        elif args.benchmark_type == 'two-player':
            run_two_player_benchmark(args)
        elif args.benchmark_type == 'training-loop':
            run_training_loop_benchmark(args)
        elif args.benchmark_type == 'simple-two-player':
            run_simple_two_player_benchmark(args)
        elif args.benchmark_type == 'model-comparison':
            run_model_comparison_benchmark(args)
        else:
            print(f"Unknown benchmark type: {args.benchmark_type}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()