#!/usr/bin/env python
"""
Regression tests for evolvepro/wrapper/process_experimental.py

This module tests the end-to-end functionality of the experimental data
processing wrapper script, including input validation, file generation,
and error handling.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest
from Bio import SeqIO


# Test data
SAMPLE_SEQUENCE = "MNTINIAKNDFS"
SAMPLE_LONG_SEQUENCE = "MNTINIAKNDFSQRWVTLPADEGHIKLMNPQ"


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_wt_fasta(temp_output_dir):
    """Create a sample wild-type FASTA file for testing."""
    fasta_path = os.path.join(temp_output_dir, "sample_WT.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">WT Wild-type sequence\n{SAMPLE_SEQUENCE}\n")
    return fasta_path


def run_wrapper_script(
    args: List[str],
    expected_returncode: int = 0,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run the wrapper script with the given arguments.

    Args:
        args: List of command-line arguments
        expected_returncode: Expected return code (default: 0)
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess object with result
    """
    cmd = [
        "pixi", "run", "-e", "evolvepro-cpu-dev",
        "python", "evolvepro/wrapper/process_experimental.py"
    ] + args

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd="/workdir/develop/EvolvePro"
    )

    assert result.returncode == expected_returncode, (
        f"Script failed with return code {result.returncode}.\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    return result


def count_fasta_sequences(fasta_path: str) -> int:
    """Count the number of sequences in a FASTA file."""
    return sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))


def get_fasta_ids(fasta_path: str) -> List[str]:
    """Get list of sequence IDs from a FASTA file."""
    return [record.id for record in SeqIO.parse(fasta_path, "fasta")]


class TestBasicFunctionality:
    """Test basic functionality of the wrapper script."""

    def test_generate_from_sequence(self, temp_output_dir):
        """Test generating WT and single mutants from a sequence string."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        # Check output files exist
        wt_fasta = os.path.join(temp_output_dir, "test_protein_WT.fasta")
        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")

        assert os.path.exists(wt_fasta), "WT FASTA file not created"
        assert os.path.exists(mutants_fasta), "Single mutants FASTA file not created"

        # Check WT sequence
        wt_records = list(SeqIO.parse(wt_fasta, "fasta"))
        assert len(wt_records) == 1, "WT FASTA should contain exactly 1 sequence"
        assert str(wt_records[0].seq) == SAMPLE_SEQUENCE, "WT sequence mismatch"
        assert wt_records[0].id == "WT", "WT sequence ID should be 'WT'"

        # Check single mutants
        # Expected: WT + (len * 19 mutants per position)
        # For 12 aa sequence: 1 WT + (12 * 19) = 229 total
        expected_count = 1 + (len(SAMPLE_SEQUENCE) * 19)
        actual_count = count_fasta_sequences(mutants_fasta)
        assert actual_count == expected_count, (
            f"Expected {expected_count} sequences, got {actual_count}"
        )

    def test_use_existing_wt_fasta(self, temp_output_dir, sample_wt_fasta):
        """Test using an existing WT FASTA file."""
        args = [
            "--wt_fasta", sample_wt_fasta,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        # Check single mutants file was created
        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")
        assert os.path.exists(mutants_fasta), "Single mutants FASTA file not created"

        # Verify mutants count
        expected_count = 1 + (len(SAMPLE_SEQUENCE) * 19)
        actual_count = count_fasta_sequences(mutants_fasta)
        assert actual_count == expected_count

    def test_verbose_output(self, temp_output_dir):
        """Test verbose logging output."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--verbose",
        ]

        result = run_wrapper_script(args)

        # Check that verbose output is present
        assert "EVOLVEpro Step 1: Experimental Data Processing" in result.stdout
        assert "Processing Complete!" in result.stdout
        assert "Generated files:" in result.stdout

    def test_quiet_mode(self, temp_output_dir):
        """Test quiet mode (without verbose flag)."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        # In quiet mode, should have minimal output (only from process.py print statement)
        assert "EVOLVEpro Step 1" not in result.stdout


class TestPositionFiltering:
    """Test position-specific mutation functionality."""

    def test_specific_positions(self, temp_output_dir):
        """Test mutating only specific positions."""
        positions = [1, 3, 5]
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--positions"
        ] + [str(p) for p in positions]

        result = run_wrapper_script(args)

        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")

        # Expected: WT + (num_positions * 19 mutants per position)
        expected_count = 1 + (len(positions) * 19)
        actual_count = count_fasta_sequences(mutants_fasta)
        assert actual_count == expected_count, (
            f"Expected {expected_count} sequences for positions {positions}, got {actual_count}"
        )

        # Verify that mutants are only at specified positions
        mutant_ids = get_fasta_ids(mutants_fasta)
        mutant_ids.remove("WT")  # Remove WT from list

        for mut_id in mutant_ids[:10]:  # Check first 10 mutants
            # Extract position from variant ID (e.g., "M1A" -> position 1)
            position = int(''.join(filter(str.isdigit, mut_id)))
            assert position in positions, (
                f"Mutant {mut_id} at position {position} not in specified positions {positions}"
            )

    def test_skip_single_mutants(self, temp_output_dir):
        """Test skipping single mutant generation."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--skip_single_mutants",
        ]

        result = run_wrapper_script(args)

        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")
        assert not os.path.exists(mutants_fasta), (
            "Single mutants file should not exist when --skip_single_mutants is set"
        )


class TestMultiMutants:
    """Test multi-mutant combination functionality."""

    @pytest.fixture
    def sample_round_file(self, temp_output_dir):
        """Create a sample experimental round Excel file."""
        import pandas as pd

        # Create sample data with variants above threshold
        data = {
            "Variant": ["M1A", "N2A", "T3A", "I4A", "N5A"],
            "activity": [1.5, 2.0, 1.2, 0.8, 1.8]
        }
        df = pd.DataFrame(data)

        excel_path = os.path.join(temp_output_dir, "round1.xlsx")
        df.to_excel(excel_path, index=False)
        return excel_path

    def test_generate_multi_mutants(self, temp_output_dir, sample_wt_fasta, sample_round_file):
        """Test generating multi-mutant combinations."""
        args = [
            "--wt_fasta", sample_wt_fasta,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--skip_single_mutants",
            "--generate_multi",
            "--multi_n", "2",
            "--multi_round_file", sample_round_file,
            "--multi_threshold", "1.0",
        ]

        result = run_wrapper_script(args)

        multi_mutants_fasta = os.path.join(temp_output_dir, "test_protein_2_mutants.fasta")
        assert os.path.exists(multi_mutants_fasta), "Multi-mutants FASTA file not created"

        # Should have WT + combinations of variants above threshold (1.5, 2.0, 1.2, 1.8)
        # That's 4 variants above threshold, so C(4,2) = 6 combinations + 1 WT = 7
        actual_count = count_fasta_sequences(multi_mutants_fasta)
        assert actual_count > 1, "Should have WT + multi-mutant combinations"

    def test_multi_mutants_without_round_file_fails(self, temp_output_dir, sample_wt_fasta):
        """Test that multi-mutant generation fails without round file."""
        args = [
            "--wt_fasta", sample_wt_fasta,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--generate_multi",
            "--multi_n", "2",
        ]

        # Should fail because --multi_round_file is required
        result = run_wrapper_script(args, expected_returncode=2, capture_output=True)
        assert "required when --generate_multi is set" in result.stderr


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_required_arguments(self):
        """Test that missing required arguments cause failure."""
        # Missing --output_dir
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "test_protein",
        ]

        result = run_wrapper_script(args, expected_returncode=2, capture_output=True)
        assert "required" in result.stderr.lower()

    def test_mutually_exclusive_wt_inputs(self, temp_output_dir, sample_wt_fasta):
        """Test that providing both wt_sequence and wt_fasta fails."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--wt_fasta", sample_wt_fasta,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args, expected_returncode=2, capture_output=True)
        assert "not allowed with argument" in result.stderr

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_output_dir = os.path.join(tmpdir, "new_subdir", "output")

            args = [
                "--wt_sequence", SAMPLE_SEQUENCE,
                "--protein_name", "test_protein",
                "--output_dir", new_output_dir,
            ]

            result = run_wrapper_script(args)

            assert os.path.exists(new_output_dir), "Output directory should be created"
            mutants_fasta = os.path.join(new_output_dir, "test_protein_single_mutants.fasta")
            assert os.path.exists(mutants_fasta)


class TestSequenceValidation:
    """Test sequence-related validations and edge cases."""

    def test_short_sequence(self, temp_output_dir):
        """Test with a very short sequence."""
        short_seq = "MNTL"
        args = [
            "--wt_sequence", short_seq,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")
        expected_count = 1 + (len(short_seq) * 19)
        actual_count = count_fasta_sequences(mutants_fasta)
        assert actual_count == expected_count

    def test_long_sequence(self, temp_output_dir):
        """Test with a longer sequence."""
        args = [
            "--wt_sequence", SAMPLE_LONG_SEQUENCE,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        mutants_fasta = os.path.join(temp_output_dir, "test_protein_single_mutants.fasta")
        expected_count = 1 + (len(SAMPLE_LONG_SEQUENCE) * 19)
        actual_count = count_fasta_sequences(mutants_fasta)
        assert actual_count == expected_count


class TestOutputFileNaming:
    """Test output file naming conventions."""

    def test_custom_protein_name(self, temp_output_dir):
        """Test that custom protein names are used in output files."""
        custom_name = "my_custom_protein"
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", custom_name,
            "--output_dir", temp_output_dir,
        ]

        result = run_wrapper_script(args)

        wt_fasta = os.path.join(temp_output_dir, f"{custom_name}_WT.fasta")
        mutants_fasta = os.path.join(temp_output_dir, f"{custom_name}_single_mutants.fasta")

        assert os.path.exists(wt_fasta)
        assert os.path.exists(mutants_fasta)

    def test_multi_mutant_naming(self, temp_output_dir):
        """Test multi-mutant file naming with different n values."""
        import pandas as pd

        # Create sample round file
        data = {
            "Variant": ["M1A", "N2A", "T3A"],
            "activity": [1.5, 2.0, 1.8]
        }
        df = pd.DataFrame(data)
        excel_path = os.path.join(temp_output_dir, "round1.xlsx")
        df.to_excel(excel_path, index=False)

        # Create WT fasta first
        wt_fasta = os.path.join(temp_output_dir, "test_WT.fasta")
        with open(wt_fasta, "w") as f:
            f.write(f">WT Wild-type sequence\n{SAMPLE_SEQUENCE}\n")

        # Test with n=3
        args = [
            "--wt_fasta", wt_fasta,
            "--protein_name", "test_protein",
            "--output_dir", temp_output_dir,
            "--skip_single_mutants",
            "--generate_multi",
            "--multi_n", "3",
            "--multi_round_file", excel_path,
            "--multi_threshold", "1.0",
        ]

        result = run_wrapper_script(args)

        multi_fasta = os.path.join(temp_output_dir, "test_protein_3_mutants.fasta")
        assert os.path.exists(multi_fasta), "Multi-mutant file with correct naming not found"


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, temp_output_dir):
        """Test a complete workflow from sequence to mutants."""
        args = [
            "--wt_sequence", SAMPLE_SEQUENCE,
            "--protein_name", "integration_test",
            "--output_dir", temp_output_dir,
            "--positions", "1", "2", "3",
            "--verbose",
        ]

        result = run_wrapper_script(args)

        # Verify all expected outputs
        wt_fasta = os.path.join(temp_output_dir, "integration_test_WT.fasta")
        mutants_fasta = os.path.join(temp_output_dir, "integration_test_single_mutants.fasta")

        assert os.path.exists(wt_fasta)
        assert os.path.exists(mutants_fasta)

        # Verify content
        wt_seq = str(list(SeqIO.parse(wt_fasta, "fasta"))[0].seq)
        assert wt_seq == SAMPLE_SEQUENCE

        # Verify verbose output
        assert "Sequence length: 12 amino acids" in result.stdout
        assert "Mutating positions: [1, 2, 3]" in result.stdout
        assert "Processing Complete!" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
