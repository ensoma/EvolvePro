#!/usr/bin/env python
"""
Regression tests for evolvepro/wrapper/dms_evolution.py

These tests verify the DMS evolution wrapper script works correctly end-to-end,
from input files to output results.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        embeddings_dir = base / "embeddings"
        labels_dir = base / "labels"
        output_dir = base / "output"

        embeddings_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        yield {
            "embeddings": embeddings_dir,
            "labels": labels_dir,
            "output": output_dir,
        }


def generate_test_embeddings(
    dataset_name: str,
    model_name: str,
    embeddings_dir: Path,
    num_variants: int = 30,
    embedding_dim: int = 50,
) -> Path:
    """
    Generate mock embeddings CSV file for testing.

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the PLM model
        embeddings_dir: Directory to save embeddings
        num_variants: Number of variants to generate
        embedding_dim: Number of embedding dimensions

    Returns:
        Path to the generated embeddings CSV file
    """
    # Generate variant names (e.g., A1G, M2L, etc.)
    variants = [f"V{i:03d}" for i in range(num_variants)]

    # Generate random embeddings
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.randn(num_variants, embedding_dim)

    # Create DataFrame
    df = pd.DataFrame(embeddings, index=variants)
    df.columns = [f"emb_{i}" for i in range(embedding_dim)]

    # Save to CSV
    output_file = embeddings_dir / f"{dataset_name}_{model_name}.csv"
    df.to_csv(output_file, index=True)

    return output_file


def generate_test_labels(
    dataset_name: str,
    labels_dir: Path,
    num_variants: int = 30,
) -> Path:
    """
    Generate mock labels CSV file for testing.

    Args:
        dataset_name: Name of the dataset
        labels_dir: Directory to save labels
        num_variants: Number of variants to generate

    Returns:
        Path to the generated labels CSV file
    """
    # Generate variant names (must match embeddings)
    variants = [f"V{i:03d}" for i in range(num_variants)]

    # Generate random activities with some structure
    np.random.seed(42)  # For reproducibility
    activities = np.random.randn(num_variants) * 2 + 5  # Mean ~5, std ~2

    # Scale activities to [0, 1]
    activities_scaled = (activities - activities.min()) / (activities.max() - activities.min())

    # Binary activities (threshold at median)
    activities_binary = (activities_scaled > 0.5).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        "variant": variants,
        "activity": activities,
        "activity_scaled": activities_scaled,
        "activity_binary": activities_binary,
    })

    # Save to CSV
    output_file = labels_dir / f"{dataset_name}_labels.csv"
    df.to_csv(output_file, index=False)

    return output_file


def run_dms_evolution(args: list, expected_success: bool = True) -> subprocess.CompletedProcess:
    """
    Run the dms_evolution.py wrapper script with the given arguments.

    Args:
        args: List of command-line arguments
        expected_success: Whether the command is expected to succeed

    Returns:
        CompletedProcess object with returncode, stdout, stderr
    """
    cmd = [
        "pixi", "run", "-e", "evolvepro-cpu-dev", "python",
        "evolvepro/wrapper/dms_evolution.py"
    ] + args

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if expected_success and result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    return result


class TestBasicFunctionality:
    """Test basic DMS evolution wrapper functionality."""

    def test_help_message(self):
        """Test that --help flag works correctly."""
        result = run_dms_evolution(["--help"])
        assert result.returncode == 0
        assert "DMS (Deep Mutational Scanning)" in result.stdout
        assert "--dataset_name" in result.stdout
        assert "--experiment_name" in result.stdout
        assert "--model_name" in result.stdout

    def test_basic_run(self, temp_dirs):
        """Test basic DMS evolution run with minimal parameters."""
        # Generate test data
        dataset_name = "test_dataset"
        model_name = "esm2_t30_150M_UR50D"
        num_variants = 30

        embeddings_file = generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=num_variants,
            embedding_dim=50,
        )

        labels_file = generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=num_variants,
        )

        # Verify test files exist
        assert embeddings_file.exists()
        assert labels_file.exists()

        # Run DMS evolution with minimal parameters
        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "basic_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "2",
            "--num_iterations", "3",
            "--num_mutants_per_round", "8",
            "--learning_strategies", "topn",
            "--regression_types", "randomforest",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify output file exists
        expected_output = temp_dirs["output"] / f"{dataset_name}_{model_name}_basic_test.csv"
        assert expected_output.exists(), f"Expected output file {expected_output} not found"

        # Verify output file has content
        df = pd.read_csv(expected_output)
        assert len(df) > 0, "Output CSV is empty"

        # Verify expected columns exist
        expected_columns = [
            "simulation_num", "round_num", "num_mutants_per_round",
            "learning_strategy", "regression_type", "first_round_strategy",
            "embedding_type", "test_error", "train_error", "spearman_corr",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not found in output"

    def test_verbose_mode(self, temp_dirs):
        """Test that verbose mode produces detailed output."""
        # Generate test data
        dataset_name = "test_verbose"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=20,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=20,
        )

        # Run with verbose flag
        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "verbose_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "1",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
            "--verbose",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify verbose output contains expected information
        assert "DMS Evolution Simulation" in result.stdout
        assert "Dataset:" in result.stdout
        assert "Simulation Parameters:" in result.stdout

    def test_quiet_mode(self, temp_dirs):
        """Test that quiet mode (no --verbose) produces minimal output."""
        # Generate test data
        dataset_name = "test_quiet"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=20,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=20,
        )

        # Run without verbose flag
        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "quiet_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "1",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Quiet mode should have much less output
        assert "DMS Evolution Simulation" not in result.stdout


class TestParameterVariations:
    """Test different parameter combinations."""

    def test_multiple_learning_strategies(self, temp_dirs):
        """Test with multiple learning strategies."""
        dataset_name = "test_strategies"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=25,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=25,
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "multi_strategy",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "1",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
            "--learning_strategies", "topn", "random",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify output contains results for both strategies
        output_file = temp_dirs["output"] / f"{dataset_name}_{model_name}_multi_strategy.csv"
        df = pd.read_csv(output_file)

        strategies = df["learning_strategy"].unique()
        assert "topn" in strategies
        assert "random" in strategies

    def test_multiple_regression_types(self, temp_dirs):
        """Test with multiple regression types."""
        dataset_name = "test_regression"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=25,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=25,
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "multi_regression",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "1",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
            "--regression_types", "randomforest", "ridge",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify output contains results for both regression types
        output_file = temp_dirs["output"] / f"{dataset_name}_{model_name}_multi_regression.csv"
        df = pd.read_csv(output_file)

        regression_types = df["regression_type"].unique()
        assert "randomforest" in regression_types
        assert "ridge" in regression_types

    def test_multiple_iterations(self, temp_dirs):
        """Test with multiple iteration counts."""
        dataset_name = "test_iterations"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=25,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=25,
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "multi_iterations",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "1",
            "--num_iterations", "2", "3",
            "--num_mutants_per_round", "5",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify output contains results for both iteration counts
        output_file = temp_dirs["output"] / f"{dataset_name}_{model_name}_multi_iterations.csv"
        df = pd.read_csv(output_file)

        # With different num_iterations, we should see different maximum round_num values
        # One simulation goes to round 2, another to round 3
        max_rounds = df["round_num"].max()
        assert max_rounds == 3  # At least one simulation ran 3 rounds


class TestErrorHandling:
    """Test error handling and validation."""

    def test_missing_required_args(self):
        """Test that missing required arguments produce errors."""
        # Missing dataset_name
        result = run_dms_evolution([
            "--experiment_name", "test",
            "--model_name", "esm2_t30_150M_UR50D",
        ], expected_success=False)
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "dataset_name" in result.stderr.lower()

    def test_invalid_learning_strategy(self, temp_dirs):
        """Test that invalid learning strategy produces error."""
        dataset_name = "test_invalid"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "invalid_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--learning_strategies", "invalid_strategy",
        ]

        result = run_dms_evolution(args, expected_success=False)
        assert result.returncode != 0
        assert "Invalid learning strategy" in result.stdout

    def test_invalid_regression_type(self, temp_dirs):
        """Test that invalid regression type produces error."""
        dataset_name = "test_invalid_reg"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "invalid_reg_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--regression_types", "invalid_regression",
        ]

        result = run_dms_evolution(args, expected_success=False)
        assert result.returncode != 0
        assert "Invalid regression type" in result.stdout

    def test_invalid_num_iterations(self, temp_dirs):
        """Test that invalid num_iterations produces error."""
        dataset_name = "test_invalid_iter"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "invalid_iter_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_iterations", "1",  # Must be > 1
        ]

        result = run_dms_evolution(args, expected_success=False)
        assert result.returncode != 0
        assert "num_iterations must be greater than 1" in result.stdout


class TestOutputValidation:
    """Test output file validation."""

    def test_output_file_format(self, temp_dirs):
        """Test that output file has correct format and columns."""
        dataset_name = "test_format"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=20,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=20,
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "format_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "2",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Load and validate output
        output_file = temp_dirs["output"] / f"{dataset_name}_{model_name}_format_test.csv"
        df = pd.read_csv(output_file)

        # Check required columns exist
        required_columns = [
            "simulation_num",
            "round_num",
            "num_mutants_per_round",
            "learning_strategy",
            "regression_type",
            "first_round_strategy",
            "embedding_type",
            "measured_var",
        ]

        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from output"

        # Check data types
        assert df["simulation_num"].dtype in [int, np.int64]
        assert df["round_num"].dtype in [int, np.int64]
        assert df["num_mutants_per_round"].dtype in [int, np.int64]
        assert df["learning_strategy"].dtype == object  # String
        assert df["regression_type"].dtype == object  # String

    def test_output_directory_creation(self, temp_dirs):
        """Test that output directory is created if it doesn't exist."""
        dataset_name = "test_dir_creation"
        model_name = "esm2_t30_150M_UR50D"

        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=20,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=20,
        )

        # Use a non-existent output directory
        new_output_dir = temp_dirs["output"] / "nested" / "output" / "dir"
        assert not new_output_dir.exists()

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "dir_test",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(new_output_dir),
            "--num_simulations", "1",
            "--num_iterations", "2",
            "--num_mutants_per_round", "5",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify directory was created
        assert new_output_dir.exists()

        # Verify output file exists
        output_file = new_output_dir / f"{dataset_name}_{model_name}_dir_test.csv"
        assert output_file.exists()


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_grid_search(self, temp_dirs):
        """Test a complete grid search with multiple parameters."""
        dataset_name = "test_grid"
        model_name = "esm2_t30_150M_UR50D"

        # Generate larger dataset for grid search
        generate_test_embeddings(
            dataset_name=dataset_name,
            model_name=model_name,
            embeddings_dir=temp_dirs["embeddings"],
            num_variants=40,
        )

        generate_test_labels(
            dataset_name=dataset_name,
            labels_dir=temp_dirs["labels"],
            num_variants=40,
        )

        args = [
            "--dataset_name", dataset_name,
            "--experiment_name", "grid_search",
            "--model_name", model_name,
            "--embeddings_path", str(temp_dirs["embeddings"]),
            "--labels_path", str(temp_dirs["labels"]),
            "--output_dir", str(temp_dirs["output"]),
            "--num_simulations", "2",
            "--num_iterations", "2", "3",
            "--num_mutants_per_round", "5", "8",
            "--learning_strategies", "topn", "random",
            "--regression_types", "randomforest", "ridge",
            "--first_round_strategies", "random",
            "--verbose",
        ]

        result = run_dms_evolution(args)
        assert result.returncode == 0

        # Verify output
        output_file = temp_dirs["output"] / f"{dataset_name}_{model_name}_grid_search.csv"
        df = pd.read_csv(output_file)

        # Should have results for multiple rounds and parameter combinations
        # With 2 simulations, 2 iterations (2, 3), 2 mutants_per_round (5, 8),
        # 2 learning_strategies, and 2 regression_types, we should have many rows
        assert len(df) > 0

        # Verify all parameter combinations are present
        assert set(df["num_mutants_per_round"].unique()) == {5, 8}
        assert set(df["learning_strategy"].unique()) == {"topn", "random"}
        assert set(df["regression_type"].unique()) == {"randomforest", "ridge"}

        # Verify we have data from multiple simulations
        assert len(df["simulation_num"].unique()) == 2
