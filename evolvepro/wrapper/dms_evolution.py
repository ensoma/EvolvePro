#!/usr/bin/env python
"""
DMS Evolution Wrapper Script

This wrapper script provides a command-line interface to run Deep Mutational Scanning (DMS)
evolution simulations using EVOLVEpro. It wraps the grid_search() function from
evolvepro.src.evolve to provide a user-friendly interface for running benchmarking and
few-shot optimization on datasets with known activities.

Usage Examples:
    # Basic DMS simulation with default parameters
    pixi run python evolvepro/wrapper/dms_evolution.py \\
      --dataset_name my_dataset \\
      --experiment_name test_run \\
      --model_name esm2_t48_15B_UR50D \\
      --embeddings_path output/plm/esm \\
      --labels_path data/dms \\
      --output_dir output/dms_results \\
      --verbose

    # Comprehensive grid search with multiple parameters
    pixi run python evolvepro/wrapper/dms_evolution.py \\
      --dataset_name brenan \\
      --experiment_name full_sweep \\
      --model_name esm2_t48_15B_UR50D \\
      --embeddings_path output/plm/esm \\
      --labels_path data/dms \\
      --num_simulations 10 \\
      --num_iterations 3 5 10 \\
      --num_mutants_per_round 8 16 32 \\
      --num_final_round_mutants 10 \\
      --learning_strategies topn random dist \\
      --regression_types randomforest ridge gradientboosting \\
      --first_round_strategies random diverse_medoids \\
      --embedding_types embeddings \\
      --measured_var activity_scaled \\
      --embeddings_file_type csv \\
      --pca_components 50 100 \\
      --output_dir output/dms_results \\
      --verbose

    # Quick test with minimal simulations
    pixi run python evolvepro/wrapper/dms_evolution.py \\
      --dataset_name test_data \\
      --experiment_name quick_test \\
      --model_name esm2_t33_650M_UR50D \\
      --embeddings_path output/plm/esm \\
      --labels_path data/dms \\
      --num_simulations 3 \\
      --num_iterations 3 \\
      --num_mutants_per_round 16 \\
      --learning_strategies topn \\
      --regression_types randomforest \\
      --output_dir output/dms_results \\
      --verbose
"""

import argparse
import logging
import sys
from typing import List, Optional

from evolvepro.src.evolve import grid_search

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: If True, set logging level to INFO; otherwise WARNING.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run DMS (Deep Mutational Scanning) evolution simulations with EVOLVEpro.",
        epilog="""
Examples:
  # Basic simulation
  %(prog)s --dataset_name my_dataset --experiment_name test \\
    --model_name esm2_t48_15B_UR50D --embeddings_path output/plm/esm \\
    --labels_path data/dms --output_dir output/dms_results --verbose

  # Full grid search
  %(prog)s --dataset_name brenan --experiment_name full_sweep \\
    --model_name esm2_t48_15B_UR50D --embeddings_path output/plm/esm \\
    --labels_path data/dms --num_simulations 10 --num_iterations 3 5 10 \\
    --learning_strategies topn random --regression_types randomforest ridge \\
    --output_dir output/dms_results --verbose

For more information, see USEAGE.md
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset (e.g., "brenan", "my_dataset")'
    )
    required.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='Name of the experiment for output files (e.g., "test_run", "full_sweep")'
    )
    required.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the PLM model used for embeddings (e.g., "esm2_t48_15B_UR50D")'
    )
    required.add_argument(
        '--embeddings_path',
        type=str,
        required=True,
        help='Path to directory containing embeddings CSV file(s)'
    )
    required.add_argument(
        '--labels_path',
        type=str,
        required=True,
        help='Path to directory containing labels CSV file(s)'
    )
    required.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )

    # Simulation parameters
    sim_params = parser.add_argument_group('simulation parameters')
    sim_params.add_argument(
        '--num_simulations',
        type=int,
        default=10,
        help='Number of independent simulations to run for statistical robustness (default: 10)'
    )
    sim_params.add_argument(
        '--num_iterations',
        type=int,
        nargs='+',
        default=[5],
        help='Number of evolution rounds (e.g., "3 5 10") (default: 5)'
    )
    sim_params.add_argument(
        '--num_mutants_per_round',
        type=int,
        nargs='+',
        default=[16],
        help='Number of variants to test per round (e.g., "8 16 32") (default: 16)'
    )
    sim_params.add_argument(
        '--num_final_round_mutants',
        type=int,
        default=10,
        help='Number of top variants to track in final round (default: 10)'
    )

    # Evolution strategy parameters
    strategy_params = parser.add_argument_group('evolution strategy parameters')
    strategy_params.add_argument(
        '--learning_strategies',
        type=str,
        nargs='+',
        default=['topn'],
        help='Learning strategies to test: topn, random, topn2bottomn2, dist (default: topn)'
    )
    strategy_params.add_argument(
        '--regression_types',
        type=str,
        nargs='+',
        default=['randomforest'],
        help='Regression models: randomforest, ridge, lasso, elasticnet, linear, '
             'neuralnet, gradientboosting, knn, gp (default: randomforest)'
    )
    strategy_params.add_argument(
        '--first_round_strategies',
        type=str,
        nargs='+',
        default=['random'],
        help='First round selection: random, diverse_medoids, representative_hie (default: random)'
    )

    # Embedding parameters
    embedding_params = parser.add_argument_group('embedding parameters')
    embedding_params.add_argument(
        '--embedding_types',
        type=str,
        nargs='+',
        default=['embeddings'],
        help='Embedding types: embeddings, embeddings_pca (default: embeddings)'
    )
    embedding_params.add_argument(
        '--embeddings_file_type',
        type=str,
        default='csv',
        choices=['csv', 'pt'],
        help='Embeddings file format: csv or pt (default: csv)'
    )
    embedding_params.add_argument(
        '--embeddings_type_pt',
        type=str,
        default=None,
        choices=['average', 'mutated', 'both'],
        help='For .pt files only: average, mutated, or both (default: None)'
    )
    embedding_params.add_argument(
        '--pca_components',
        type=int,
        nargs='*',
        default=None,
        help='PCA dimensionality reduction components (e.g., "50 100 200") (default: None)'
    )

    # Activity measurement parameters
    activity_params = parser.add_argument_group('activity parameters')
    activity_params.add_argument(
        '--measured_var',
        type=str,
        nargs='+',
        default=['activity_scaled'],
        help='Activity measurement column(s): activity, activity_scaled, activity_binary '
             '(default: activity_scaled)'
    )

    # General options
    general = parser.add_argument_group('general options')
    general.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress information'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If arguments are invalid.
    """
    # Validate num_iterations
    if any(n <= 1 for n in args.num_iterations):
        raise ValueError("All values in --num_iterations must be greater than 1")

    # Validate num_simulations
    if args.num_simulations <= 0:
        raise ValueError("--num_simulations must be greater than 0")

    # Validate num_final_round_mutants
    if args.num_final_round_mutants <= 0:
        raise ValueError("--num_final_round_mutants must be greater than 0")

    # Validate num_mutants_per_round
    if any(n <= 0 for n in args.num_mutants_per_round):
        raise ValueError("All values in --num_mutants_per_round must be greater than 0")

    # Validate learning strategies
    valid_learning_strategies = ['topn', 'random', 'topn2bottomn2', 'dist']
    for strategy in args.learning_strategies:
        if strategy not in valid_learning_strategies:
            raise ValueError(
                f"Invalid learning strategy: {strategy}. "
                f"Valid options: {', '.join(valid_learning_strategies)}"
            )

    # Validate regression types
    valid_regression_types = [
        'ridge', 'lasso', 'elasticnet', 'linear', 'neuralnet',
        'randomforest', 'gradientboosting', 'knn', 'gp'
    ]
    for reg_type in args.regression_types:
        if reg_type not in valid_regression_types:
            raise ValueError(
                f"Invalid regression type: {reg_type}. "
                f"Valid options: {', '.join(valid_regression_types)}"
            )

    # Validate first round strategies
    valid_first_round = ['random', 'diverse_medoids', 'representative_hie']
    for strategy in args.first_round_strategies:
        if strategy not in valid_first_round:
            raise ValueError(
                f"Invalid first round strategy: {strategy}. "
                f"Valid options: {', '.join(valid_first_round)}"
            )

    # Validate measured_var
    valid_measured_vars = ['activity', 'activity_scaled', 'activity_binary']
    for var in args.measured_var:
        if var not in valid_measured_vars:
            raise ValueError(
                f"Invalid measured variable: {var}. "
                f"Valid options: {', '.join(valid_measured_vars)}"
            )

    # Warn about PCA components
    if args.pca_components and 'embeddings_pca' not in args.embedding_types:
        logger.warning(
            "PCA components specified but 'embeddings_pca' not in --embedding_types. "
            "Consider adding 'embeddings_pca' to use PCA-reduced embeddings."
        )


def main() -> None:
    """
    Main execution function.

    Parses arguments, validates them, and runs the DMS evolution simulation.
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    # Log start
    logger.info("=" * 80)
    logger.info("DMS Evolution Simulation")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Embeddings path: {args.embeddings_path}")
    logger.info(f"Labels path: {args.labels_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Validate arguments
    try:
        validate_arguments(args)
    except ValueError as e:
        logger.error(f"Argument validation error: {e}")
        sys.exit(1)

    # Log simulation parameters
    logger.info("Simulation Parameters:")
    logger.info(f"  Number of simulations: {args.num_simulations}")
    logger.info(f"  Number of iterations: {args.num_iterations}")
    logger.info(f"  Mutants per round: {args.num_mutants_per_round}")
    logger.info(f"  Final round mutants: {args.num_final_round_mutants}")
    logger.info("")
    logger.info("Strategy Parameters:")
    logger.info(f"  Learning strategies: {', '.join(args.learning_strategies)}")
    logger.info(f"  Regression types: {', '.join(args.regression_types)}")
    logger.info(f"  First round strategies: {', '.join(args.first_round_strategies)}")
    logger.info("")
    logger.info("Embedding Parameters:")
    logger.info(f"  Embedding types: {', '.join(args.embedding_types)}")
    logger.info(f"  Embeddings file type: {args.embeddings_file_type}")
    if args.pca_components:
        logger.info(f"  PCA components: {args.pca_components}")
    if args.embeddings_type_pt:
        logger.info(f"  PT embeddings type: {args.embeddings_type_pt}")
    logger.info("")
    logger.info("Activity Parameters:")
    logger.info(f"  Measured variables: {', '.join(args.measured_var)}")
    logger.info("")
    logger.info("Starting grid search...")
    logger.info("=" * 80)
    logger.info("")

    # Run grid search
    try:
        grid_search(
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            model_name=args.model_name,
            embeddings_path=args.embeddings_path,
            labels_path=args.labels_path,
            num_simulations=args.num_simulations,
            num_iterations=args.num_iterations,
            measured_var=args.measured_var,
            learning_strategies=args.learning_strategies,
            num_mutants_per_round=args.num_mutants_per_round,
            num_final_round_mutants=args.num_final_round_mutants,
            first_round_strategies=args.first_round_strategies,
            embedding_types=args.embedding_types,
            pca_components=args.pca_components,
            regression_types=args.regression_types,
            embeddings_file_type=args.embeddings_file_type,
            output_dir=args.output_dir,
            embeddings_type_pt=args.embeddings_type_pt,
        )
        logger.info("")
        logger.info("=" * 80)
        logger.info("DMS evolution simulation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error during grid search: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
