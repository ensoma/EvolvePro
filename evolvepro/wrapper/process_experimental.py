#!/usr/bin/env python
"""
EVOLVEpro Step 1: Experimental Data Processing Wrapper

This script provides a command-line interface for processing experimental protein data,
including generating wild-type FASTA files, single amino acid mutants, and optional
multi-mutant combinations.

Usage:
    python evolvepro/wrapper/process_experimental.py \
        --wt_sequence MNTINIAKNDFS \
        --protein_name my_protein \
        --output_dir output/exp

    python evolvepro/wrapper/process_experimental.py \
        --wt_fasta data/my_protein_WT.fasta \
        --protein_name my_protein \
        --output_dir output/exp \
        --positions 1 5 10 15 \
        --generate_multi \
        --multi_n 2 \
        --multi_round_file data/exp/round1.xlsx \
        --multi_threshold 1.0
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add the parent directory to the path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evolvepro.src.process import (
    generate_wt,
    generate_single_aa_mutants,
    generate_n_mutant_combinations,
)

# Configure logging
logger = logging.getLogger(__name__)


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
        description="EVOLVEpro Step 1: Process experimental protein data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with sequence string:
  python evolvepro/wrapper/process_experimental.py \\
      --wt_sequence MNTINIAKNDFSQRWVTLP \\
      --protein_name my_protein \\
      --output_dir output/exp

  # Use existing WT FASTA and specify positions:
  python evolvepro/wrapper/process_experimental.py \\
      --wt_fasta data/my_protein_WT.fasta \\
      --protein_name my_protein \\
      --output_dir output/exp \\
      --positions 1 5 10 15

  # Generate multi-mutants:
  python evolvepro/wrapper/process_experimental.py \\
      --wt_fasta output/exp/my_protein_WT.fasta \\
      --protein_name my_protein \\
      --output_dir output/exp \\
      --generate_multi \\
      --multi_n 2 \\
      --multi_round_file data/exp/round1.xlsx \\
      --multi_threshold 1.0
        """,
    )

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--protein_name",
        type=str,
        required=True,
        help="Name of the protein (used for output file naming)",
    )
    required.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output FASTA files",
    )

    # Wild-type sequence input (mutually exclusive)
    wt_group = parser.add_mutually_exclusive_group(required=True)
    wt_group.add_argument(
        "--wt_sequence",
        type=str,
        help="Wild-type protein sequence as a string (e.g., MNTINIAKNDFS)",
    )
    wt_group.add_argument(
        "--wt_fasta",
        type=str,
        help="Path to existing wild-type FASTA file",
    )

    # Single mutant generation options
    single_mutant_group = parser.add_argument_group("single mutant options")
    single_mutant_group.add_argument(
        "--skip_single_mutants",
        action="store_true",
        help="Skip generation of single amino acid mutants",
    )
    single_mutant_group.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help="Specific positions to mutate (1-based indexing). If not specified, all positions are mutated.",
    )

    # Multi-mutant generation options
    multi_mutant_group = parser.add_argument_group("multi-mutant options")
    multi_mutant_group.add_argument(
        "--generate_multi",
        action="store_true",
        help="Generate multi-mutant combinations",
    )
    multi_mutant_group.add_argument(
        "--multi_n",
        type=int,
        default=2,
        help="Number of mutations to combine for multi-mutants (default: 2)",
    )
    multi_mutant_group.add_argument(
        "--multi_round_file",
        type=str,
        help="Path to Excel file with experimental round data (required if --generate_multi is set)",
    )
    multi_mutant_group.add_argument(
        "--multi_threshold",
        type=float,
        default=1.0,
        help="Minimum activity threshold to include a variant in multi-mutant combinations (default: 1.0)",
    )

    # Output control
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    args = parser.parse_args()

    # Validation: if generate_multi is set, multi_round_file must be provided
    if args.generate_multi and not args.multi_round_file:
        parser.error("--multi_round_file is required when --generate_multi is set")

    return args


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    # Setup logging based on verbosity
    setup_logging(args.verbose)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("EVOLVEpro Step 1: Experimental Data Processing")
    logger.info("=" * 60)
    logger.info(f"Protein name: {args.protein_name}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Define output file paths
    wt_fasta_path = os.path.join(args.output_dir, f"{args.protein_name}_WT.fasta")
    single_mutants_path = os.path.join(
        args.output_dir, f"{args.protein_name}_single_mutants.fasta"
    )
    multi_mutants_path = os.path.join(
        args.output_dir, f"{args.protein_name}_{args.multi_n}_mutants.fasta"
    )

    # Step 1: Generate or use wild-type FASTA
    if args.wt_sequence:
        logger.info("[1/3] Generating wild-type FASTA from sequence...")
        logger.info(f"      Sequence length: {len(args.wt_sequence)} amino acids")
        logger.info(f"      Output: {wt_fasta_path}")

        generate_wt(args.wt_sequence, wt_fasta_path)

        logger.info("      ✓ Wild-type FASTA generated")
        logger.info("")
    else:
        logger.info(f"[1/3] Using existing wild-type FASTA: {args.wt_fasta}")
        logger.info("")

        # Copy or reference the existing WT FASTA
        wt_fasta_path = args.wt_fasta

    # Step 2: Generate single amino acid mutants
    if not args.skip_single_mutants:
        logger.info("[2/3] Generating single amino acid mutants...")
        if args.positions:
            logger.info(f"      Mutating positions: {args.positions}")
        else:
            logger.info("      Mutating all positions")
        logger.info(f"      Output: {single_mutants_path}")

        generate_single_aa_mutants(
            wt_fasta=wt_fasta_path,
            output_file=single_mutants_path,
            positions=args.positions,
        )

        logger.info("      ✓ Single mutants generated")
        logger.info("")
    else:
        logger.info("[2/3] Skipping single mutant generation (--skip_single_mutants)")
        logger.info("")

    # Step 3: Generate multi-mutant combinations (optional)
    if args.generate_multi:
        logger.info(f"[3/3] Generating {args.multi_n}-mutant combinations...")
        logger.info(f"      Round file: {args.multi_round_file}")
        logger.info(f"      Threshold: {args.multi_threshold}")
        logger.info(f"      Output: {multi_mutants_path}")

        try:
            generate_n_mutant_combinations(
                wt_fasta=wt_fasta_path,
                mutant_file=args.multi_round_file,
                n=args.multi_n,
                output_file=multi_mutants_path,
                threshold=args.multi_threshold,
            )

            logger.info("      ✓ Multi-mutants generated")
            logger.info("")
        except Exception as e:
            logger.error(f"      ✗ Error generating multi-mutants: {e}")
            logger.warning("      Continuing without multi-mutant generation...")
            logger.info("")
    else:
        logger.info("[3/3] Skipping multi-mutant generation")
        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("Processing Complete!")
    logger.info("=" * 60)
    logger.info("Generated files:")
    if args.wt_sequence:
        logger.info(f"  - Wild-type FASTA: {wt_fasta_path}")
    if not args.skip_single_mutants:
        logger.info(f"  - Single mutants: {single_mutants_path}")
    if args.generate_multi:
        logger.info(f"  - {args.multi_n}-mutants: {multi_mutants_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Activate the PLM environment: conda activate plm")
    logger.info("  2. Extract embeddings using Step 2 (PLM extraction)")
    logger.info("     Example:")
    logger.info(f"     python evolvepro/plm/esm/extract.py \\")
    logger.info(f"       esm2_t48_15B_UR50D \\")
    logger.info(f"       {single_mutants_path} \\")
    logger.info(f"       {args.output_dir}/embeddings \\")
    logger.info(f"       --toks_per_batch 512 \\")
    logger.info(f"       --include mean \\")
    logger.info(f"       --concatenate_dir {args.output_dir}/")
    logger.info("")


if __name__ == "__main__":
    main()
