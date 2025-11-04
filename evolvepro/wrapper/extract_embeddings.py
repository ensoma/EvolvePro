#!/usr/bin/env python
"""
EVOLVEpro Step 2: ESM Embedding Extraction Wrapper

This script provides a command-line interface for extracting ESM (Evolutionary Scale Modeling)
embeddings from protein FASTA files. ESM models are state-of-the-art protein language models
developed by Meta AI.

By default, ESM models are downloaded to a 'models' directory adjacent to the output directory.
You can customize the model cache location using the --model_cache_dir option.

Usage:
    pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
        --fasta_file output/exp/my_protein_single_mutants.fasta \
        --output_dir output/exp/embeddings \
        --protein_name my_protein \
        --model esm2_t48_15B_UR50D

    pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
        --fasta_file data/my_sequences.fasta \
        --output_dir output/embeddings \
        --protein_name my_protein \
        --model esm2_t33_650M_UR50D \
        --gpu \
        --verbose
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configure logging
logger = logging.getLogger(__name__)


# Supported ESM models and their properties
SUPPORTED_MODELS = {
    "esm2_t48_15B_UR50D": {
        "description": "ESM-2 15B parameter model (best performance, requires more memory)",
        "default_batch": 512,
    },
    "esm2_t36_3B_UR50D": {
        "description": "ESM-2 3B parameter model (good balance)",
        "default_batch": 1024,
    },
    "esm2_t33_650M_UR50D": {
        "description": "ESM-2 650M parameter model (smaller, faster)",
        "default_batch": 2048,
    },
    "esm2_t30_150M_UR50D": {
        "description": "ESM-2 150M parameter model (fastest)",
        "default_batch": 4096,
    },
    "esm1b_t33_650M_UR50S": {
        "description": "ESM-1b 650M parameter model",
        "default_batch": 2048,
    },
}


def setup_logging(verbose: bool) -> None:
    """
    Configure logging based on verbosity level.

    Args:
        verbose: If True, set logging level to INFO, otherwise WARNING
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
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="EVOLVEpro Step 2: Extract ESM embeddings from protein sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with ESM-2 15B model:
  pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \\
      --fasta_file output/exp/my_protein_single_mutants.fasta \\
      --output_dir output/exp/embeddings \\
      --protein_name my_protein \\
      --model esm2_t48_15B_UR50D

  # Using smaller/faster model with GPU:
  pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \\
      --fasta_file data/sequences.fasta \\
      --output_dir output/embeddings \\
      --protein_name my_protein \\
      --model esm2_t33_650M_UR50D \\
      --gpu \\
      --verbose

  # Custom batch size:
  pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \\
      --fasta_file data/sequences.fasta \\
      --output_dir output/embeddings \\
      --protein_name my_protein \\
      --model esm2_t48_15B_UR50D \\
      --toks_per_batch 256

  # Custom model cache directory:
  pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \\
      --fasta_file data/sequences.fasta \\
      --output_dir output/embeddings \\
      --protein_name my_protein \\
      --model esm2_t33_650M_UR50D \\
      --model_cache_dir /path/to/model/cache

  # List available models:
  pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py --list_models
        """,
    )

    # Special action for listing models
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all available PLM models and exit",
    )

    # Required arguments (only if not listing models)
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--fasta_file",
        type=str,
        help="Path to input FASTA file with protein sequences",
    )
    required.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save extracted embeddings",
    )
    required.add_argument(
        "--protein_name",
        type=str,
        help="Name of the protein (used for output CSV file naming)",
    )
    required.add_argument(
        "--model",
        type=str,
        choices=list(SUPPORTED_MODELS.keys()),
        help="ESM model to use for embedding extraction",
    )

    # Optional arguments
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded ESM models. If not specified, defaults to '{output_dir}/models'.",
    )
    optional.add_argument(
        "--toks_per_batch",
        type=int,
        default=None,
        help="Maximum batch size (tokens per batch). If not specified, uses model-specific default.",
    )
    optional.add_argument(
        "--repr_layers",
        type=int,
        nargs="+",
        default=[-1],
        help="Layer indices from which to extract representations (default: -1 for last layer)",
    )
    optional.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=["mean"],
        choices=["mean", "per_tok", "bos", "contacts"],
        help="Which representations to extract (default: mean)",
    )
    optional.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022,
        help="Truncate sequences longer than this value (default: 1022)",
    )

    # Hardware options
    hardware = parser.add_argument_group("hardware options")
    hardware.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available (default: CPU only)",
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--no_concatenate",
        action="store_true",
        help="Do not concatenate individual .pt files into a single CSV (default: concatenate)",
    )
    output.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # If listing models, we don't need other arguments
    if args.list_models:
        return args

    # Validate required arguments
    if not args.fasta_file or not args.output_dir or not args.protein_name or not args.model:
        parser.error(
            "the following arguments are required: --fasta_file, --output_dir, --protein_name, --model\n"
            "Use --list_models to see available models."
        )

    return args


def list_models() -> None:
    """Print all available ESM models."""
    logger.info("=" * 80)
    logger.info("Available ESM Models")
    logger.info("=" * 80)
    logger.info("")

    for model_name, model_info in SUPPORTED_MODELS.items():
        logger.info(f"  • {model_name}")
        logger.info(f"    {model_info['description']}")
        logger.info(f"    Default batch size: {model_info['default_batch']}")
        logger.info("")

    logger.info("=" * 80)


def run_esm_extraction(
    model_name: str,
    fasta_file: str,
    output_dir: str,
    protein_name: str,
    toks_per_batch: int,
    repr_layers: List[int],
    include: List[str],
    truncation_seq_length: int,
    use_gpu: bool,
    concatenate: bool,
    model_cache_dir: str,
) -> int:
    """
    Run the ESM extraction script.

    Args:
        model_name: ESM model name
        fasta_file: Path to input FASTA file
        output_dir: Output directory for embeddings
        protein_name: Protein name for output files
        toks_per_batch: Batch size
        repr_layers: Layer indices to extract
        include: Which representations to include
        truncation_seq_length: Sequence truncation length
        use_gpu: Whether to use GPU
        concatenate: Whether to concatenate outputs
        model_cache_dir: Directory to cache downloaded models

    Returns:
        Return code from the extraction script
    """
    # Construct path to ESM extraction script
    esm_script = Path(__file__).resolve().parents[1] / "plm" / "esm" / "extract.py"

    if not esm_script.exists():
        logger.error(f"ESM extraction script not found: {esm_script}")
        return 1

    # Build command
    cmd = [
        "python",
        str(esm_script),
        model_name,
        fasta_file,
        output_dir,
        "--toks_per_batch", str(toks_per_batch),
        "--repr_layers"
    ] + [str(layer) for layer in repr_layers] + [
        "--include"
    ] + include + [
        "--truncation_seq_length", str(truncation_seq_length),
    ]

    # Add GPU flag if not using GPU (ESM script defaults to GPU if available)
    if not use_gpu:
        cmd.append("--nogpu")

    # Add concatenation directory if requested
    if concatenate:
        concatenate_dir = str(Path(output_dir).parent)
        cmd.extend(["--concatenate_dir", concatenate_dir])

    # Set up environment to control model cache location
    env = os.environ.copy()
    env["TORCH_HOME"] = model_cache_dir

    logger.info(f"Running ESM extraction...")
    logger.info(f"Model cache directory: {model_cache_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")

    # Run the extraction script with custom environment
    result = subprocess.run(cmd, env=env)

    return result.returncode


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    # Handle --list_models (must be before setup_logging)
    if args.list_models:
        # For list_models, always show output
        setup_logging(True)
        list_models()
        return

    # Setup logging for normal operation
    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("EVOLVEpro Step 2: ESM Embedding Extraction")
    logger.info("=" * 80)
    logger.info(f"FASTA file: {args.fasta_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Protein name: {args.protein_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: {'Enabled' if args.gpu else 'Disabled (CPU only)'}")
    logger.info("")

    # Validate input file exists
    if not os.path.exists(args.fasta_file):
        logger.error(f"FASTA file not found: {args.fasta_file}")
        logger.error("Please check the file path and try again.")
        sys.exit(1)

    # Get model info
    model_info = SUPPORTED_MODELS[args.model]

    # Use default batch size if not specified
    toks_per_batch = args.toks_per_batch or model_info["default_batch"]

    # Set model cache directory (default to output_dir/models if not specified)
    if args.model_cache_dir:
        model_cache_dir = args.model_cache_dir
    else:
        model_cache_dir = str(Path(args.output_dir).parent / "models")

    logger.info(f"Model description: {model_info['description']}")
    logger.info(f"Tokens per batch: {toks_per_batch}")
    logger.info(f"Representation layers: {args.repr_layers}")
    logger.info(f"Include: {args.include}")
    logger.info(f"Concatenate outputs: {not args.no_concatenate}")
    logger.info(f"Model cache directory: {model_cache_dir}")
    logger.info("")

    # Create output and model cache directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    # Run extraction
    logger.info("Starting embedding extraction...")
    logger.info("=" * 80)
    logger.info("")

    returncode = run_esm_extraction(
        model_name=args.model,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        protein_name=args.protein_name,
        toks_per_batch=toks_per_batch,
        repr_layers=args.repr_layers,
        include=args.include,
        truncation_seq_length=args.truncation_seq_length,
        use_gpu=args.gpu,
        concatenate=not args.no_concatenate,
        model_cache_dir=model_cache_dir,
    )

    if returncode != 0:
        logger.error("")
        logger.error("=" * 80)
        logger.error("Embedding extraction failed!")
        logger.error("=" * 80)
        sys.exit(returncode)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Embedding Extraction Complete!")
    logger.info("=" * 80)

    if not args.no_concatenate:
        # Find the generated CSV file
        fasta_filename = Path(args.fasta_file).stem
        csv_file = Path(args.output_dir).parent / f"{fasta_filename}_{args.model}.csv"
        if csv_file.exists():
            logger.info(f"Embeddings CSV: {csv_file}")
        else:
            logger.warning("Expected CSV file not found. Check output directory.")

    logger.info(f"Raw embeddings directory: {args.output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Use the embeddings CSV file in Step 3 (Evolution)")
    logger.info("  - For experimental workflow: Run evolvepro.src.evolve.evolve_experimental()")
    logger.info("  - For DMS workflow: Run scripts/dms/dms_main.py")
    logger.info("")


if __name__ == "__main__":
    main()
