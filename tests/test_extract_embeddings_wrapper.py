#!/usr/bin/env python
"""
Regression tests for evolvepro/wrapper/extract_embeddings.py

This module tests the end-to-end functionality of the ESM embedding extraction
wrapper script, including model selection, output generation, and error handling.

NOTE: These tests use the smallest ESM model (esm2_t30_150M_UR50D) to minimize
download time and computational requirements.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import pytest
from Bio import SeqIO


# Test data - short protein sequences to speed up testing
SAMPLE_SEQUENCES = {
    "WT": "MNTINIAKNDFS",
    "M1A": "ANTINIAKNDFS",
    "N2A": "MATINIAKNDFS",
}

# Use the smallest/fastest ESM model for testing
TEST_MODEL = "esm2_t30_150M_UR50D"


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_fasta(temp_output_dir):
    """Create a sample FASTA file with test sequences."""
    fasta_path = os.path.join(temp_output_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        for seq_id, sequence in SAMPLE_SEQUENCES.items():
            f.write(f">{seq_id}\n{sequence}\n")
    return fasta_path


@pytest.fixture
def shared_model_cache(tmp_path_factory):
    """Create a shared model cache directory to avoid re-downloading models."""
    cache_dir = tmp_path_factory.mktemp("model_cache")
    return str(cache_dir)


def run_wrapper_script(
    args: List[str],
    expected_returncode: int = 0,
    capture_output: bool = True,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """
    Run the wrapper script with the given arguments.

    Args:
        args: List of command-line arguments
        expected_returncode: Expected return code (default: 0)
        capture_output: Whether to capture stdout/stderr
        timeout: Timeout in seconds (default: 300s = 5min)

    Returns:
        CompletedProcess object with result
    """
    cmd = [
        "pixi", "run", "-e", "plm-cpu-dev",
        "python", "evolvepro/wrapper/extract_embeddings.py"
    ] + args

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        cwd="/workdir/develop/EvolvePro",
        timeout=timeout,
    )

    assert result.returncode == expected_returncode, (
        f"Script failed with return code {result.returncode}.\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )

    return result


class TestListModels:
    """Test --list_models functionality."""

    def test_list_models(self):
        """Test that --list_models displays available models."""
        args = ["--list_models"]
        result = run_wrapper_script(args)

        # Check that all supported models are listed
        assert "esm2_t48_15B_UR50D" in result.stdout
        assert "esm2_t36_3B_UR50D" in result.stdout
        assert "esm2_t33_650M_UR50D" in result.stdout
        assert "esm2_t30_150M_UR50D" in result.stdout
        assert "esm1b_t33_650M_UR50S" in result.stdout

        # Check that descriptions are present
        assert "Available ESM Models" in result.stdout
        assert "Default batch size" in result.stdout


class TestBasicFunctionality:
    """Test basic embedding extraction functionality."""

    def test_basic_extraction(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test basic embedding extraction from FASTA file."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
        ]

        result = run_wrapper_script(args, timeout=600)

        # Check that embeddings directory was created
        assert os.path.exists(embeddings_dir), "Embeddings directory not created"

        # Check that .pt files were created (one per sequence)
        pt_files = list(Path(embeddings_dir).glob("*.pt"))
        assert len(pt_files) == len(SAMPLE_SEQUENCES), (
            f"Expected {len(SAMPLE_SEQUENCES)} .pt files, found {len(pt_files)}"
        )

        # Check that CSV file was created (concatenated embeddings)
        csv_file = Path(temp_output_dir) / f"test_sequences_{TEST_MODEL}.csv"
        assert csv_file.exists(), f"Expected CSV file not found: {csv_file}"

    def test_verbose_output(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test verbose logging output."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Check that verbose output is present
        assert "EVOLVEpro Step 2: ESM Embedding Extraction" in result.stdout
        assert "FASTA file:" in result.stdout
        assert "Output directory:" in result.stdout
        assert "Model:" in result.stdout
        assert "Embedding Extraction Complete!" in result.stdout

    def test_quiet_mode(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test quiet mode (without verbose flag)."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
        ]

        result = run_wrapper_script(args, timeout=600)

        # In quiet mode, wrapper's info messages should not be present
        assert "EVOLVEpro Step 2" not in result.stdout


class TestModelOptions:
    """Test model selection and configuration options."""

    def test_custom_batch_size(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test custom batch size configuration."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")
        custom_batch = 1024

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--toks_per_batch", str(custom_batch),
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Verify that custom batch size is used in command
        assert f"Tokens per batch: {custom_batch}" in result.stdout

    def test_custom_model_cache_dir(self, temp_output_dir, sample_fasta):
        """Test custom model cache directory."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")
        custom_cache = os.path.join(temp_output_dir, "custom_cache")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", custom_cache,
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Verify that custom cache directory was created
        assert os.path.exists(custom_cache), "Custom model cache directory not created"
        assert f"Model cache directory: {custom_cache}" in result.stdout


class TestHardwareOptions:
    """Test GPU/CPU hardware options."""

    def test_cpu_mode(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test CPU-only mode (default)."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Verify CPU mode is indicated
        assert "GPU: Disabled (CPU only)" in result.stdout
        # Check that --nogpu flag is in the command
        assert "--nogpu" in result.stdout

    def test_gpu_flag(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test GPU flag (even if GPU is not available)."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--gpu",
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Verify GPU mode is indicated
        assert "GPU: Enabled" in result.stdout


class TestOutputOptions:
    """Test output file generation options."""

    def test_no_concatenate(self, temp_output_dir, sample_fasta, shared_model_cache):
        """Test disabling CSV concatenation."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--no_concatenate",
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Check that .pt files were created
        pt_files = list(Path(embeddings_dir).glob("*.pt"))
        assert len(pt_files) == len(SAMPLE_SEQUENCES)

        # Check that CSV file was NOT created
        csv_file = Path(temp_output_dir) / f"test_sequences_{TEST_MODEL}.csv"
        assert not csv_file.exists(), "CSV file should not exist with --no_concatenate"

        # Verify output option in verbose log
        assert "Concatenate outputs: False" in result.stdout

    def test_output_directory_creation(self, sample_fasta, shared_model_cache):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_output = os.path.join(tmpdir, "nested", "output", "embeddings")

            args = [
                "--fasta_file", sample_fasta,
                "--output_dir", nested_output,
                "--protein_name", "test_protein",
                "--model", TEST_MODEL,
                "--model_cache_dir", shared_model_cache,
            ]

            result = run_wrapper_script(args, timeout=600)

            assert os.path.exists(nested_output), "Nested output directory not created"
            # Check that files were created in the nested directory
            pt_files = list(Path(nested_output).glob("*.pt"))
            assert len(pt_files) > 0, "No .pt files created in nested directory"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_required_arguments(self):
        """Test that missing required arguments cause failure."""
        # Missing all required arguments
        args = []
        result = run_wrapper_script(args, expected_returncode=2, capture_output=True)
        assert "required" in result.stderr.lower()

    def test_missing_fasta_file(self, temp_output_dir, shared_model_cache):
        """Test handling of non-existent FASTA file."""
        nonexistent_fasta = os.path.join(temp_output_dir, "nonexistent.fasta")
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", nonexistent_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
        ]

        result = run_wrapper_script(args, expected_returncode=1, capture_output=True)
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_invalid_model(self, temp_output_dir, sample_fasta):
        """Test handling of invalid model name."""
        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", sample_fasta,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", "invalid_model_name",
        ]

        result = run_wrapper_script(args, expected_returncode=2, capture_output=True)
        assert "invalid choice" in result.stderr.lower()


class TestSequenceValidation:
    """Test with different sequence types and edge cases."""

    def test_single_sequence(self, temp_output_dir, shared_model_cache):
        """Test with a single sequence FASTA file."""
        fasta_path = os.path.join(temp_output_dir, "single.fasta")
        with open(fasta_path, "w") as f:
            f.write(">WT\nMNTINIAKNDFS\n")

        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", fasta_path,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
        ]

        result = run_wrapper_script(args, timeout=600)

        # Check that exactly one .pt file was created
        pt_files = list(Path(embeddings_dir).glob("*.pt"))
        assert len(pt_files) == 1, "Expected exactly 1 .pt file for single sequence"

    def test_short_sequences(self, temp_output_dir, shared_model_cache):
        """Test with very short protein sequences."""
        fasta_path = os.path.join(temp_output_dir, "short.fasta")
        with open(fasta_path, "w") as f:
            f.write(">short1\nMNTL\n")
            f.write(">short2\nAKND\n")

        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", fasta_path,
            "--output_dir", embeddings_dir,
            "--protein_name", "test_protein",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
        ]

        result = run_wrapper_script(args, timeout=600)

        # Check that .pt files were created
        pt_files = list(Path(embeddings_dir).glob("*.pt"))
        assert len(pt_files) == 2, "Expected 2 .pt files for 2 short sequences"


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, temp_output_dir, shared_model_cache):
        """Test a complete workflow from FASTA to embeddings CSV."""
        # Create test FASTA
        fasta_path = os.path.join(temp_output_dir, "workflow_test.fasta")
        with open(fasta_path, "w") as f:
            f.write(">WT\nMNTINIAKNDFS\n")
            f.write(">M1A\nANTINIAKNDFS\n")
            f.write(">N2A\nMATINIAKNDFS\n")

        embeddings_dir = os.path.join(temp_output_dir, "embeddings")

        args = [
            "--fasta_file", fasta_path,
            "--output_dir", embeddings_dir,
            "--protein_name", "workflow_test",
            "--model", TEST_MODEL,
            "--model_cache_dir", shared_model_cache,
            "--verbose",
        ]

        result = run_wrapper_script(args, timeout=600)

        # Verify all outputs
        assert os.path.exists(embeddings_dir)

        # Check .pt files
        pt_files = list(Path(embeddings_dir).glob("*.pt"))
        assert len(pt_files) == 3, "Expected 3 .pt files"

        # Check CSV file
        csv_file = Path(temp_output_dir) / f"workflow_test_{TEST_MODEL}.csv"
        assert csv_file.exists(), "CSV file not created"

        # Verify verbose output includes next steps
        assert "Next steps:" in result.stdout
        assert "Use the embeddings CSV file in Step 3" in result.stdout
        assert "Embedding Extraction Complete!" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
