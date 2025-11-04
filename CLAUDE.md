# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EVOLVEpro is a protein engineering platform that uses protein language model (PLM) embeddings with a lightweight random forest regression model to optimize protein properties through iterative active learning. The system can evolve proteins with as few as 10 experimental data points per round, enabling multi-objective evolution campaigns.

## Environment Management

The project uses **two separate conda environments** to avoid dependency conflicts:

### 1. EVOLVEpro Environment (Main)
Used for all core functionality (processing, running models, visualization).

**Setup with conda:**
```bash
conda env create -f environment.yml
conda activate evolvepro
```

**Setup with Pixi (alternative, CPU-only):**
```bash
pixi install
pixi shell
```

### 2. PLM Environment
Used only for extracting protein language model embeddings.

**Setup with conda:**
```bash
sh setup_plm.sh
conda activate plm
```

**Setup with Pixi (alternative, CPU-only):**
```bash
pixi install -e plm-cpu
pixi shell -e plm-cpu
```

**Important:**
- Always verify which environment is needed before running commands. Most development work uses the `evolvepro` environment.
- Pixi environments were added after forking the repository to improve software environment management. Currently, both pixi environments (default workspace and plm-cpu) provide CPU-only support. GPU support may be added in the future.

## Core Architecture

The codebase follows a 4-step pipeline workflow:

### 1. Process (`evolvepro/src/process.py`)
- Generates FASTA files and CSV files with variant sequences
- Creates single amino acid mutants or n-mutant combinations
- Processes DMS (Deep Mutational Scanning) datasets

### 2. PLM Embedding Extraction (`evolvepro/plm/`)
- Extracts embeddings from various protein language models (ESM, ProtT5, UniRep, Ankh, ProteinBERT)
- Each PLM has its own subdirectory with an `extract.py` script
- Outputs CSV files of embeddings from FASTA input

### 3. Evolution (`evolvepro/src/evolve.py` + `evolvepro/src/model.py`)
Two main workflows:
- **DMS workflow**: Few-shot optimization on datasets with known activities
- **Experimental workflow**: Iterative experimental optimization across rounds

The evolution process:
- `evolve.py`: Orchestrates the directed evolution simulation
- `model.py`: Implements first-round selection strategies and top-layer regression models
- Supports multiple regression types: ridge, lasso, elasticnet, linear, neuralnet, randomforest, gradientboosting, knn, gp
- Learning strategies: random, topn, topn2bottomn2, dist
- First-round strategies: random, diverse_medoids, representative_hie, explicit_variants

### 4. Visualization (`evolvepro/src/plot.py`)
- DMS benchmark plotting functions
- Experimental round tracking and visualization

## Key Source Files

- `evolvepro/src/data.py`: Loading and aligning embeddings with labels
- `evolvepro/src/utils.py`: Utility functions including PCA transformations
- `evolvepro/src/process.py`: Data preprocessing and FASTA generation (core functions)
- `evolvepro/src/evolve.py`: Main simulation orchestration
- `evolvepro/src/model.py`: Regression models and selection strategies
- `evolvepro/src/plot.py`: Visualization utilities
- `evolvepro/wrapper/process_experimental.py`: Command-line wrapper for experimental data processing (Step 1)
- `evolvepro/wrapper/extract_embeddings.py`: Command-line wrapper for ESM embedding extraction (Step 2)

## Common Development Commands

### Running DMS Workflow
```bash
conda activate evolvepro
python scripts/dms/dms_main.py \
  --dataset_name brenan \
  --experiment_name test_run \
  --model_name esm2_t48_15B_UR50D \
  --embeddings_path output/plm/esm \
  --labels_path data/dms \
  --num_simulations 10 \
  --num_iterations 5 \
  --learning_strategies topn \
  --regression_types randomforest
```

### Running Experimental Workflow
Create a Python script (e.g., `scripts/exp/my_protein.py`):
```python
from evolvepro.src.evolve import evolve_experimental

evolve_experimental(
    protein_name='my_protein',
    round_name='Round2',
    embeddings_base_path='/path/to/embeddings',
    embeddings_file_name='embeddings.csv',
    round_base_path='/path/to/rounds',
    round_file_names=['Round1.xlsx', 'Round2.xlsx'],
    wt_fasta_path='/path/to/wt.fasta',
    rename_WT=True,
    number_of_variants=12,
    output_dir='/path/to/output'
)
```

Then run:
```bash
conda activate evolvepro
python scripts/exp/my_protein.py
```

### Extracting ESM Embeddings (Wrapper Script)

The easiest way to extract ESM embeddings is using the wrapper script:

**Basic usage:**
```bash
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file output/exp/my_protein_single_mutants.fasta \
  --output_dir output/exp/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --verbose
```

**Using a smaller/faster model:**
```bash
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file data/sequences.fasta \
  --output_dir output/embeddings \
  --protein_name my_protein \
  --model esm2_t33_650M_UR50D \
  --verbose
```

**With GPU acceleration:**
```bash
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file data/sequences.fasta \
  --output_dir output/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --gpu \
  --verbose
```

**List available models:**
```bash
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py --list_models
```

**Using conda instead of pixi:**
```bash
conda activate plm
python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file output/exp/my_protein_single_mutants.fasta \
  --output_dir output/exp/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --verbose
```

### Extracting PLM Embeddings (Direct API)

For advanced users or other PLM models, use the extraction scripts directly:

```bash
conda activate plm
python evolvepro/plm/esm/extract.py \
  esm2_t48_15B_UR50D \
  output/dms/dataset.fasta \
  output/plm/esm/dataset \
  --toks_per_batch 512 \
  --include mean \
  --concatenate_dir output/plm/esm/
```

### Processing Experimental Data (Wrapper Script)

The easiest way to process experimental data is using the wrapper script:

**Basic usage with a protein sequence:**
```bash
pixi run python evolvepro/wrapper/process_experimental.py \
  --wt_sequence MNTINIAKNDFSQRWVTLP \
  --protein_name my_protein \
  --output_dir output/exp \
  --verbose
```

**Using an existing WT FASTA file:**
```bash
pixi run python evolvepro/wrapper/process_experimental.py \
  --wt_fasta data/exp/my_protein_WT.fasta \
  --protein_name my_protein \
  --output_dir output/exp \
  --verbose
```

**Mutating specific positions only:**
```bash
pixi run python evolvepro/wrapper/process_experimental.py \
  --wt_sequence MNTINIAKNDFSQRWVTLP \
  --protein_name my_protein \
  --output_dir output/exp \
  --positions 1 5 10 15 20 \
  --verbose
```

**Generating multi-mutant combinations:**
```bash
pixi run python evolvepro/wrapper/process_experimental.py \
  --wt_fasta output/exp/my_protein_WT.fasta \
  --protein_name my_protein \
  --output_dir output/exp \
  --generate_multi \
  --multi_n 2 \
  --multi_round_file data/exp/round1.xlsx \
  --multi_threshold 1.0 \
  --verbose
```

**Using conda instead of pixi:**
```bash
conda activate evolvepro
python evolvepro/wrapper/process_experimental.py \
  --wt_sequence MNTINIAKNDFSQRWVTLP \
  --protein_name my_protein \
  --output_dir output/exp \
  --verbose
```

### Processing Data (Programmatic API)

For more control, you can use the Python API directly:

```python
from evolvepro.src.process import generate_wt, generate_single_aa_mutants

# Generate wild-type and single mutants
generate_wt('MNTINIAKNDFS', 'output/dataset_WT.fasta')
generate_single_aa_mutants('output/dataset_WT.fasta', 'output/dataset.fasta')
```

## Available Wrapper Scripts

Wrapper scripts in `evolvepro/wrapper/` provide user-friendly command-line interfaces to EVOLVEpro functionality:

### process_experimental.py

**Purpose**: Step 1 of the experimental workflow - generates FASTA files for variant sequences

**Key Features**:
- Generate wild-type FASTA from a sequence string or use existing FASTA file
- Generate all single amino acid mutants or mutants at specific positions
- Optional: Generate multi-mutant combinations from previous experimental rounds
- Verbose logging mode for detailed progress tracking
- CPU-only (no GPU required)

**Common Options**:
- `--wt_sequence`: Provide protein sequence as a string
- `--wt_fasta`: Use an existing WT FASTA file
- `--protein_name`: Name for output files (required)
- `--output_dir`: Output directory (required)
- `--positions`: Specific positions to mutate (1-based indexing)
- `--skip_single_mutants`: Skip single mutant generation
- `--generate_multi`: Enable multi-mutant combinations
- `--multi_n`: Number of mutations to combine (default: 2)
- `--multi_round_file`: Excel file with experimental round data
- `--multi_threshold`: Activity threshold for multi-mutants (default: 1.0)
- `--verbose`: Show detailed progress information
- `--gpu`: GPU flag (included for consistency, but Step 1 is CPU-only)

**Output Files**:
- `{protein_name}_WT.fasta`: Wild-type sequence (if generated from string)
- `{protein_name}_single_mutants.fasta`: All single amino acid mutants
- `{protein_name}_{n}_mutants.fasta`: Multi-mutant combinations (if requested)

**Example Workflow**:
```bash
# Step 1: Generate variants
pixi run python evolvepro/wrapper/process_experimental.py \
  --wt_sequence MNTINIAKNDFSQRWVTLP \
  --protein_name my_protein \
  --output_dir output/exp \
  --verbose

# Step 2: Extract embeddings (using wrapper)
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file output/exp/my_protein_single_mutants.fasta \
  --output_dir output/exp/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --verbose

# Step 3: Run experimental evolution (back to default environment)
pixi run python scripts/exp/my_protein.py
```

### extract_embeddings.py

**Purpose**: Step 2 of the experimental workflow - extracts ESM embeddings from protein FASTA files

**Key Features**:
- Extract embeddings using ESM models (ESM-2 and ESM-1b)
- Automatic model downloading and caching
- Configurable batch sizes optimized for different models
- GPU and CPU support
- Automatic concatenation of embeddings into CSV format
- Verbose logging mode for detailed progress tracking
- List available models with `--list_models`

**Common Options**:
- `--fasta_file`: Path to input FASTA file (required)
- `--output_dir`: Output directory for embeddings (required)
- `--protein_name`: Protein name for output files (required)
- `--model`: ESM model to use (required, see available models below)
- `--toks_per_batch`: Batch size (auto-optimized per model if not specified)
- `--repr_layers`: Layer indices to extract (default: -1 for last layer)
- `--include`: Representations to extract (default: mean)
- `--gpu`: Enable GPU acceleration (default: CPU only)
- `--no_concatenate`: Skip CSV concatenation (default: concatenate)
- `--verbose`: Show detailed progress information
- `--list_models`: List all available ESM models

**Available ESM Models**:
- `esm2_t48_15B_UR50D`: ESM-2 15B (best performance, more memory)
- `esm2_t36_3B_UR50D`: ESM-2 3B (good balance)
- `esm2_t33_650M_UR50D`: ESM-2 650M (smaller, faster)
- `esm2_t30_150M_UR50D`: ESM-2 150M (fastest)
- `esm1b_t33_650M_UR50S`: ESM-1b 650M

**Output Files**:
- `{output_dir}/*.pt`: Individual embedding files (one per sequence)
- `{fasta_filename}_{model_name}.csv`: Concatenated embeddings CSV (default)

**Example Usage**:
```bash
# List available models
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py --list_models

# Basic usage with ESM-2 15B
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file output/exp/my_protein_single_mutants.fasta \
  --output_dir output/exp/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --verbose

# Using smaller model for faster processing
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file data/sequences.fasta \
  --output_dir output/embeddings \
  --protein_name my_protein \
  --model esm2_t33_650M_UR50D \
  --verbose

# With GPU acceleration (if available)
pixi run -e plm-cpu python evolvepro/wrapper/extract_embeddings.py \
  --fasta_file data/sequences.fasta \
  --output_dir output/embeddings \
  --protein_name my_protein \
  --model esm2_t48_15B_UR50D \
  --gpu \
  --verbose
```

## Project Structure Notes

- `scripts/`: Contains workflow-specific example scripts organized by task (dms, exp, plm, plot, process)
  - SLURM scripts (*.sh) are provided for HPC environments but can be adapted for local command-line use
- `evolvepro/src/`: Core library code
- `evolvepro/plm/`: PLM-specific extraction scripts (each model in its own subdirectory)
- `evolvepro/wrapper/`: Command-line wrapper scripts providing user-friendly interfaces to core functionality
- `tests/`: Pytest regression tests for wrapper scripts and core functionality
- `data/`: Input data organized by workflow (dms/, exp/)
- `colab/`: Google Colab tutorial notebook

## Wrapper Scripts Coding Standards

Scripts in the `evolvepro/wrapper/` directory provide command-line interfaces to core EVOLVEpro functionality. All wrapper scripts MUST adhere to the following standards:

### Required Features

1. **Logging**: Use Python's `logging` module instead of `print()` statements
   - Configure logging with a `setup_logging()` function that accepts a verbosity parameter
   - Use appropriate log levels: `logger.info()` for normal output, `logger.warning()` for warnings, `logger.error()` for errors
   - Format: Simple message format without timestamps (e.g., `format='%(message)s'`)
   - Example:
     ```python
     import logging

     logger = logging.getLogger(__name__)

     def setup_logging(verbose: bool) -> None:
         level = logging.INFO if verbose else logging.WARNING
         logging.basicConfig(
             level=level,
             format='%(message)s',
             handlers=[logging.StreamHandler(sys.stdout)]
         )
     ```

2. **Type Hints**: Add type hints to all function definitions
   - Use `typing` module for complex types (`Optional`, `List`, etc.)
   - Include return type hints (use `-> None` for functions that don't return values)
   - Example:
     ```python
     from typing import Optional, List

     def parse_arguments() -> argparse.Namespace:
         """Parse command-line arguments."""
         ...

     def main() -> None:
         """Main execution function."""
         ...
     ```

3. **Command-line Arguments**: Use `argparse` with organized argument groups
   - Group related arguments (e.g., "required arguments", "single mutant options")
   - Provide clear help text for all arguments
   - Include usage examples in the epilog

4. **Documentation**: Include comprehensive docstrings
   - Module-level docstring with usage examples
   - Function docstrings with Args and Returns sections

### Example Wrapper Script Structure

```python
#!/usr/bin/env python
"""
Brief description of the wrapper script.

Usage examples go here.
"""

import argparse
import logging
import sys
from typing import Optional, List

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(...)
    # Add arguments
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logger.info("Processing started...")
    # Main logic here


if __name__ == "__main__":
    main()
```

## Testing

EVOLVEpro uses pytest for testing. Tests are located in the `tests/` directory.

### Setting Up the Test Environment

The project includes a dedicated development environment with testing dependencies:

**Install the development environment:**
```bash
pixi install -e evolvepro-cpu-dev
```

This environment includes:
- pytest: Testing framework
- ruff: Linting and formatting
- pyright: Type checking
- All EVOLVEpro dependencies

### Running Tests

**Run all tests:**
```bash
pixi run -e evolvepro-cpu-dev pytest
```

**Run specific test file:**
```bash
pixi run -e evolvepro-cpu-dev pytest tests/test_process_experimental_wrapper.py
```

**Run with verbose output:**
```bash
pixi run -e evolvepro-cpu-dev pytest -v
```

**Run specific test class or function:**
```bash
pixi run -e evolvepro-cpu-dev pytest tests/test_process_experimental_wrapper.py::TestBasicFunctionality
pixi run -e evolvepro-cpu-dev pytest tests/test_process_experimental_wrapper.py::TestBasicFunctionality::test_generate_from_sequence
```

**Run with coverage (if pytest-cov is installed):**
```bash
pixi run -e evolvepro-cpu-dev pytest --cov=evolvepro --cov-report=html
```

### Available Tests

#### test_process_experimental_wrapper.py

Comprehensive regression tests for the `evolvepro/wrapper/process_experimental.py` wrapper script.

**Test Coverage:**
- **TestBasicFunctionality**: Core functionality tests
  - Generating WT and mutants from sequence string
  - Using existing WT FASTA files
  - Verbose and quiet output modes

- **TestPositionFiltering**: Position-specific mutation tests
  - Mutating specific positions only
  - Skipping single mutant generation

- **TestMultiMutants**: Multi-mutant combination tests
  - Generating multi-mutant combinations
  - Threshold filtering
  - Error handling for missing round files

- **TestErrorHandling**: Error cases and validation
  - Missing required arguments
  - Mutually exclusive arguments
  - Output directory creation

- **TestSequenceValidation**: Sequence edge cases
  - Short sequences
  - Long sequences

- **TestOutputFileNaming**: Output file naming conventions
  - Custom protein names
  - Multi-mutant file naming

- **TestIntegration**: End-to-end workflow tests
  - Complete workflows from input to output

**Running wrapper tests:**
```bash
pixi run -e evolvepro-cpu-dev pytest tests/test_process_experimental_wrapper.py -v
```

### Writing New Tests

When adding new wrapper scripts or functionality, follow these testing guidelines:

1. **Create test files** in the `tests/` directory with the naming pattern `test_*.py`
2. **Use fixtures** for common setup (temporary directories, sample files, etc.)
3. **Test end-to-end** functionality, not just individual functions
4. **Test error cases** as well as success cases
5. **Verify outputs** by checking file existence, content, and format
6. **Use descriptive test names** that explain what is being tested
7. **Organize tests** into classes by functionality (e.g., `TestBasicFunctionality`, `TestErrorHandling`)

**Example test structure:**
```python
import pytest
import tempfile
import subprocess

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

class TestNewFeature:
    """Tests for new feature."""

    def test_basic_functionality(self, temp_output_dir):
        """Test that basic feature works correctly."""
        # Arrange
        args = ["--option", "value"]

        # Act
        result = run_script(args)

        # Assert
        assert result.returncode == 0
        assert os.path.exists(expected_output)
```

### Pytest Configuration

Pytest is configured in `pyproject.toml`:
- Test directory: `tests/`
- Test file pattern: `test_*.py`
- Test class pattern: `Test*`
- Test function pattern: `test_*`

## Important Constraints

- **NumPy version**: The main `evolvepro` environment requires `numpy<2.0` due to `scikit-learn-extra` compatibility
- **Python version**: PLM environment locked to Python 3.10 for compatibility with protein language models
- **GPU requirements**: PLM embedding extraction is designed for GPU/HPC environments. Adjust `--toks_per_batch` based on GPU memory.
- **Separate environments**: Never mix conda environments. PLM extraction requires the `plm` environment; all other operations use `evolvepro`.

## Data Flow

### Experimental Workflow (Recommended)
1. **Input**: Wild-type protein sequence (string or FASTA file)
2. **Step 1 - Process** (use `evolvepro/wrapper/process_experimental.py`):
   - Generate wild-type FASTA (if needed)
   - Generate single amino acid mutants or multi-mutant combinations
   - Output: FASTA file(s) of all variants
3. **Step 2 - ESM Embedding Extraction** (use `evolvepro/wrapper/extract_embeddings.py`, requires PLM environment):
   - Extract embeddings using ESM models
   - Output: CSV file of embeddings (one row per variant)
4. **Step 3 - Evolution** (use `evolvepro.src.evolve.evolve_experimental`):
   - Load embeddings and experimental round data (Excel files with activity measurements)
   - Train regression model to predict variant fitness
   - Rank variants and select top candidates for next round
   - Output: Predicted top variants + visualizations

### DMS Workflow
1. **Input**: Protein sequence(s) in FASTA format
2. **Process**: Generate variants → FASTA file of all variants
3. **PLM**: Extract embeddings → CSV file of embeddings (one row per variant)
4. **Evolve**: Simulate evolution with known activities from DMS datasets
5. **Output**: Performance metrics and visualizations

## File Formats

- **FASTA files**: Protein sequences (wild-type and variants)
- **Embeddings**: CSV files with variant names as index, embedding dimensions as columns
- **Labels**: CSV files with columns: variant, activity, activity_scaled, activity_binary
- **Experimental rounds**: Excel files with variant names and measured activities
- **Output**: CSV files with predictions, rankings, and model performance metrics
