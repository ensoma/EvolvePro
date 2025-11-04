# EVOLVEpro Usage Guide

## Overview

EVOLVEpro is a protein engineering platform that combines protein language model (PLM) embeddings with machine learning regression models to optimize protein properties through iterative active learning. This guide provides detailed instructions for running the software, including inputs, analysis steps, outputs, and optional configurations.

## Workflow Architecture

EVOLVEpro follows a **4-step pipeline**:

1. **Process**: Generate variant sequences and FASTA files
2. **PLM Embedding Extraction**: Extract numerical representations from protein sequences
3. **Evolution**: Train ML models and predict optimal variants
4. **Visualization**: Analyze and plot results

## Two Main Workflows

### 1. DMS (Deep Mutational Scanning) Workflow
Benchmarking and few-shot optimization on datasets with known activities. Used for validating the approach or simulating evolution campaigns on characterized datasets.

### 2. Experimental Workflow
Iterative experimental optimization across rounds. Used for real protein engineering campaigns where variants are tested in the lab.

---

## Step 1: Process (Data Preparation)

**Purpose**: Generate variant sequences, create FASTA files, and prepare labels for downstream analysis.

**Script Location**: `evolvepro/src/process.py`

**Environment**: `evolvepro` (main environment)

### For DMS Data

**Input**:
- Raw DMS data (Excel `.xlsx` or CSV `.csv` file) with columns for variants and activity measurements
- Wild-type sequence in FASTA format

**Key Functions**:
- `process_dataset()`: Process DMS datasets, apply activity cutoffs, and generate labels
- `generate_mutation_fasta()`: Create FASTA file of mutant sequences

**Example Usage**:
```python
from evolvepro.src.process import process_dataset

filtered_df, fractions = process_dataset(
    file_path='data/dms/my_dataset.xlsx',
    wt_fasta_path='data/dms/WT.fasta',
    dataset_name='my_dataset',
    activity_column='fitness',
    cutoff_value=1.0,
    output_dir='output/dms',
    sheet_name=None,  # Optional: specify Excel sheet name
    cutoff_rule='greater_than',  # Options: 'greater_than', 'less_than', 'custom'
    cutoff_percentiles=[50, 75, 90],  # Optional: additional percentile cutoffs
    AA_shift=None,  # Optional: adjust amino acid position numbering
    drop_columns=True  # Keep only essential columns
)
```

**Outputs**:
- `{dataset_name}.fasta`: FASTA file containing all variant sequences
- `{dataset_name}_labels.csv`: CSV file with columns:
  - `variant`: Variant name (e.g., "A123G", "WT")
  - `activity`: Raw activity values
  - `activity_scaled`: Min-max normalized activities (0-1)
  - `activity_binary`: Binary classification (0/1) based on cutoff

### For Experimental Data

**Input**:
- Wild-type protein sequence (as string or FASTA file)

**Key Functions**:
- `generate_wt()`: Create wild-type FASTA file
- `generate_single_aa_mutants()`: Generate all single amino acid mutations
- `generate_n_mutant_combinations()`: Create multi-mutant combinations

**Example Usage**:
```python
from evolvepro.src.process import generate_wt, generate_single_aa_mutants

# Step 1: Generate wild-type FASTA
wt_sequence = 'MNTINIAKNDFS'
generate_wt(wt_sequence, 'output/exp/my_protein_WT.fasta')

# Step 2: Generate all single mutants
generate_single_aa_mutants(
    wt_fasta='output/exp/my_protein_WT.fasta',
    output_file='output/exp/my_protein_single_mutants.fasta',
    positions=None  # Optional: list of positions to mutate (e.g., [1, 5, 10])
)

# Optional Step 3: Generate multi-mutant combinations
generate_n_mutant_combinations(
    wt_fasta='output/exp/my_protein_WT.fasta',
    mutant_file='data/exp/round1_results.xlsx',
    n=2,  # Number of mutations to combine
    output_file='output/exp/my_protein_double_mutants.fasta',
    threshold=1.0  # Minimum activity to include a variant
)
```

**Outputs**:
- FASTA file(s) containing variant sequences ready for embedding extraction

**Additional Utilities**:
- `suggest_initial_mutants()`: Randomly select variants for initial experimental testing
- `plot_mutations_per_position()`: Visualize mutation distribution
- `plot_histogram_of_readout()`: Plot activity distributions

---

## Step 2: PLM Embedding Extraction

**Purpose**: Convert protein sequences into numerical vector representations using pre-trained protein language models.

**Script Location**: `evolvepro/plm/{model_name}/extract.py`

**Environment**: `plm` (separate environment required)

### Supported Models

Each model has its own subdirectory in `evolvepro/plm/`:
- **ESM** (esm2_t48_15B_UR50D, esm2_t36_3B_UR50D, etc.)
- **ProtT5** (prot_t5_xl_uniref50, prot_t5_xl_bfd)
- **UniRep** (unirep)
- **Ankh** (ankh_base, ankh_large)
- **ProteinBERT** (proteinbert)
- **One-hot encoding** (simple baseline)

### ESM Model (Most Common)

**Input**:
- FASTA file containing protein sequences

**Command**:
```bash
python evolvepro/plm/esm/extract.py \
  esm2_t48_15B_UR50D \
  output/dms/my_dataset.fasta \
  output/plm/esm/my_dataset \
  --toks_per_batch 512 \
  --include mean \
  --repr_layers -1 \
  --concatenate_dir output/plm/esm/
```

**Parameters**:
- **First positional arg**: Model name or path to model file
- **Second positional arg**: Input FASTA file path
- **Third positional arg**: Output directory for individual `.pt` files
- `--toks_per_batch`: Batch size (tokens per batch). **GPU memory dependent**:
  - Small GPU (8GB): 512
  - Medium GPU (16GB): 1024
  - Large GPU (32GB): 2048-4096
- `--include`: Representation type(s) to extract:
  - `mean`: Mean-pooled sequence representation (recommended for EVOLVEpro)
  - `per_tok`: Per-residue representations
  - `bos`: Beginning-of-sequence token
  - `contacts`: Contact predictions
- `--repr_layers`: Which model layer(s) to extract (default: `-1` for final layer)
- `--concatenate_dir`: Directory to save concatenated CSV file (required for EVOLVEpro)
- `--nogpu`: Force CPU-only mode (not recommended for large datasets)

**GPU Acceleration**:
- **Automatic GPU detection**: The script automatically uses GPU if available (see line 83-85 in `extract.py`)
- **To enable GPU**: Ensure PyTorch is installed with CUDA support and CUDA is available
- **To disable GPU**: Add `--nogpu` flag
- **Check GPU usage**: Script prints "Transferred model to GPU" and shows device info

**Outputs**:
- Individual `.pt` files: One PyTorch file per sequence in `output_dir/`
- **Concatenated CSV**: `{fasta_name}_{model_name}.csv` in `concatenate_dir/`
  - **This CSV is the input for Step 3**
  - Format: Rows = variants, Columns = embedding dimensions
  - Index column = variant names

### Other PLM Models

**ProtT5**:
```bash
python evolvepro/plm/prot_t5/extract.py \
  prot_t5_xl_uniref50 \
  output/dms/my_dataset.fasta \
  output/plm/prot_t5/my_dataset \
  --include mean \
  --concatenate_dir output/plm/prot_t5/
```

**Ankh**:
```bash
python evolvepro/plm/ankh/extract.py \
  ankh_base \
  output/dms/my_dataset.fasta \
  output/plm/ankh/my_dataset \
  --include mean \
  --concatenate_dir output/plm/ankh/
```

**Important Notes**:
- Always activate the `plm` environment before running extraction scripts
- GPU is highly recommended for large models (ESM-15B, ProtT5)
- For very long sequences, adjust `--truncation_seq_length` parameter
- Each model produces different embedding dimensions:
  - ESM2-15B: 5120 dimensions
  - ESM2-3B: 2560 dimensions
  - ProtT5: 1024 dimensions

---

## Step 3: Evolution (Model Training & Prediction)

**Purpose**: Train machine learning models on experimental data and predict high-performing variants for the next round.

**Script Location**:
- DMS: `scripts/dms/dms_main.py` or use `evolvepro.src.evolve.grid_search()`
- Experimental: Custom scripts (see `scripts/exp/*.py`) using `evolvepro.src.evolve.evolve_experimental()`

**Environment**: `evolvepro` (main environment)

### A. DMS Workflow

**Input**:
- Embeddings CSV file (from Step 2)
- Labels CSV file (from Step 1)

**Command-Line Interface**:
```bash
python scripts/dms/dms_main.py \
  --dataset_name my_dataset \
  --experiment_name test_run \
  --model_name esm2_t48_15B_UR50D \
  --embeddings_path output/plm/esm \
  --labels_path data/dms \
  --num_simulations 10 \
  --num_iterations 5 \
  --num_mutants_per_round 10 16 \
  --num_final_round_mutants 10 \
  --learning_strategies topn random \
  --regression_types randomforest ridge \
  --first_round_strategies random diverse_medoids \
  --embedding_types embeddings \
  --measured_var activity_scaled \
  --embeddings_file_type csv \
  --pca_components 50 100 \
  --output_dir output/dms_results
```

**Key Parameters**:

**Simulation Settings**:
- `--num_simulations`: Number of independent simulation runs (for statistical robustness)
- `--num_iterations`: Number of evolution rounds (e.g., 5 = 5 rounds of selection)
- `--num_mutants_per_round`: How many variants to test per round
- `--num_final_round_mutants`: How many top variants to track in final round

**Learning Strategies** (`--learning_strategies`):
- `topn`: Select top-N predicted variants (greedy exploitation)
- `random`: Random selection (baseline)
- `topn2bottomn2`: Split between top and bottom predictions (exploration)
- `dist`: Distance-based diversity selection

**Regression Models** (`--regression_types`):
- `randomforest`: Random Forest (default, robust)
- `ridge`: Ridge regression (L2 regularization)
- `lasso`: Lasso regression (L1 regularization)
- `elasticnet`: Elastic Net (L1+L2)
- `linear`: Ordinary least squares
- `neuralnet`: Neural network (MLP)
- `gradientboosting`: XGBoost
- `knn`: K-nearest neighbors
- `gp`: Gaussian Process

**First Round Strategies** (`--first_round_strategies`):
- `random`: Random initial selection
- `diverse_medoids`: K-medoids clustering for diversity
- `explicit_variants`: Manually specify starting variants (requires code modification)

**Embedding Types** (`--embedding_types`):
- `embeddings`: Full embeddings
- `embeddings_pca_50`: PCA-reduced embeddings (50 components)
- Automatically generated if `--pca_components` is specified

**Activity Measures** (`--measured_var`):
- `activity`: Raw activity values
- `activity_scaled`: Min-max scaled (0-1)
- `activity_binary`: Binary classification (0/1)

**Output**:
- CSV file: `{dataset_name}_{model_name}_{experiment_name}.csv`
- Contains per-round metrics for all parameter combinations:
  - Prediction errors (train/test MSE)
  - R² scores
  - Top variant identities
  - Spearman correlations
  - Binary classification accuracy

### B. Experimental Workflow

**Input**:
- Embeddings CSV file (all variants)
- Excel files with experimental measurements for each round
- Wild-type FASTA file

**Python Script Example**:
```python
from evolvepro.src.evolve import evolve_experimental

protein_name = 'my_protein'
round_name = 'Round2'

this_round_variants, df_test, df_sorted_all = evolve_experimental(
    protein_name=protein_name,
    round_name=round_name,
    embeddings_base_path='output/plm/esm',
    embeddings_file_name='my_protein_esm2_t48_15B_UR50D.csv',
    round_base_path='data/exp/rounds',
    round_file_names=['Round1.xlsx', 'Round2.xlsx'],  # All rounds tested so far
    wt_fasta_path='data/exp/my_protein_WT.fasta',
    rename_WT=True,  # Standardize WT naming
    number_of_variants=12,  # Number of top predictions to display
    output_dir='output/exp_results'
)

# Display top predictions
print(f"\nTop {number_of_variants} variants for next round:")
print(df_test.sort_values(by='y_pred', ascending=False).head(number_of_variants))
```

**Round File Format** (Excel):
Required columns:
- `variant`: Variant name (e.g., "A123G", "WT")
- `activity`: Measured activity value

Optional columns:
- Any metadata (will be preserved)

**Iterative Usage**:
```python
# Round 1: Initial random variants tested
round_file_names = ['Round1.xlsx']

# Round 2: Train on Round1, predict for Round2
round_file_names = ['Round1.xlsx', 'Round2.xlsx']

# Round 3: Train on Round1+2, predict for Round3
round_file_names = ['Round1.xlsx', 'Round2.xlsx', 'Round3.xlsx']

# Can exclude problematic rounds:
round_file_names = ['Round1.xlsx', 'Round4.xlsx']  # Skip Round2-3
```

**Output Files** (saved to `{output_dir}/{protein_name}/{round_name}/`):
- `iteration.csv`: Iteration assignments for all variants
- `this_round_variants.csv`: Variants tested in current round
- `df_test.csv`: Predictions for all untested variants (sorted by `y_pred`)
- `df_sorted_all.csv`: Comprehensive results for all variants

**Model Configuration**:
The experimental workflow currently uses:
- **Regression model**: Random Forest (hardcoded in `evolve_experimental()` at line 481)
- **To change model**: Modify `regression_type='randomforest'` parameter

**Optional: Multi-Mutant Support**:
```python
from evolvepro.src.evolve import evolve_experimental_multi

# For combining single and multi-mutant libraries
this_round_variants, df_test, df_sorted_all = evolve_experimental_multi(
    protein_name=protein_name,
    round_name=round_name,
    embeddings_base_path='output/plm/esm',
    embeddings_file_names=['single_mutants.csv', 'double_mutants.csv'],
    round_base_path='data/exp/rounds',
    round_file_names_single=['Round1.xlsx'],
    round_file_names_multi=['Round2_double.xlsx'],
    wt_fasta_path='data/exp/my_protein_WT.fasta',
    rename_WT=True,
    number_of_variants=12,
    output_dir='output/exp_results'
)
```

---

## Step 4: Visualization (Optional)

**Purpose**: Generate plots to analyze evolution performance and track progress.

**Script Location**: `evolvepro/src/plot.py`

**Environment**: `evolvepro` (main environment)

### DMS Benchmark Plotting

**Input**:
- Results CSV from DMS workflow (Step 3A)

**Functions** (examples in `scripts/plot/dms.py`):
- Plot performance metrics across rounds
- Compare different learning strategies
- Visualize convergence to optimal variants

### Experimental Round Tracking

**Input**:
- Round result files from experimental workflow (Step 3B)

**Functions** (examples in `scripts/plot/exp.py`):
- Track activity improvements across rounds
- Visualize prediction accuracy
- Compare predicted vs. actual activities

---

## GPU Acceleration

### Where GPU is Used

**Step 2 (PLM Embedding Extraction)**:
- **Critical for performance**: Embedding extraction is GPU-accelerated
- **Automatic detection**: Scripts automatically use GPU if available
- **GPU memory requirements**:
  - ESM2-15B: ~10GB GPU memory minimum
  - ESM2-3B: ~4GB GPU memory
  - ProtT5-XL: ~12GB GPU memory

**How to Enable GPU**:
1. Install PyTorch with CUDA support:
   ```bash
   # Check CUDA version
   nvidia-smi

   # Install PyTorch with matching CUDA (example for CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify GPU availability:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True
   print(torch.cuda.get_device_name(0))  # Shows GPU name
   ```

3. The PLM extraction scripts will automatically use GPU (no code changes needed)

**Monitoring GPU Usage**:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**GPU Memory Optimization**:
Adjust `--toks_per_batch` parameter based on GPU memory:
```bash
# Small GPU (8GB VRAM)
--toks_per_batch 256

# Medium GPU (16GB VRAM)
--toks_per_batch 512

# Large GPU (32GB+ VRAM)
--toks_per_batch 2048
```

**CPU-Only Mode**:
If no GPU is available or you want to force CPU:
```bash
python evolvepro/plm/esm/extract.py \
  esm2_t48_15B_UR50D \
  input.fasta \
  output_dir \
  --toks_per_batch 128 \
  --include mean \
  --nogpu \
  --concatenate_dir output/
```
  **Warning**: CPU-only mode is significantly slower (10-100x) for large models

### Steps NOT Using GPU

**Step 1 (Process)**: Pure data processing, no GPU benefit

**Step 3 (Evolution)**:
- Current implementation uses CPU-based scikit-learn models
- Random Forest, Ridge, etc. are optimized for CPU
- GPU would provide minimal benefit for small-scale regression

**Step 4 (Visualization)**: Plotting is CPU-based

---

## Alternative and Optional Steps

### 1. Alternative Embedding Methods

**PCA Dimensionality Reduction**:
```python
from evolvepro.src.utils import pca_embeddings

# Reduce embeddings to N components
embeddings_pca = pca_embeddings(embeddings, n_components=50)
```
- **Purpose**: Reduce overfitting, speed up training
- **Recommended for**: Small datasets (<100 sequences)
- **Specify in DMS workflow**: `--pca_components 50 100 200`

**One-Hot Encoding** (Baseline):
```bash
python evolvepro/plm/one-hot/extract.py \
  input.fasta \
  output_dir \
  --concatenate_dir output/
```
- **Purpose**: Simple baseline without pre-trained models
- **Use case**: Quick prototyping or comparison

### 2. Alternative First-Round Selection

**Diverse Medoids** (K-medoids clustering):
- **Purpose**: Select maximally diverse initial variants
- **Requires**: `scikit-learn-extra` package
- **Enable**: `--first_round_strategies diverse_medoids`
- **Note**: Automatically performs PCA for computational efficiency

**Explicit Variants**:
```python
# In custom script
explicit_variants = ['A123G', 'V45I', 'T78A']

output_table = directed_evolution_simulation(
    labels=labels,
    embeddings=embeddings,
    first_round_strategy='explicit_variants',
    explicit_variants=explicit_variants,
    ...
)
```

### 3. Custom Activity Cutoffs

**Multiple Percentile Cutoffs**:
```python
filtered_df, fractions = process_dataset(
    file_path='data.xlsx',
    cutoff_value=1.0,
    cutoff_percentiles=[50, 75, 90, 95],  # Generate multiple binary classifications
    cutoff_rule='greater_than',
    ...
)
# Creates: activity_binary, activity_binary_50p, activity_binary_75p, etc.
```

**Custom Cutoff Function**:
```python
def custom_cutoff(df, activity_column, cutoff):
    # Example: Combine activity and p-value
    return (df[activity_column] > cutoff) & (df['p_value'] < 0.05)

filtered_df, fractions = process_dataset(
    cutoff_rule='custom',
    cutoff_function=custom_cutoff,
    ...
)
```

### 4. Grid Search Parameter Sweeps

**Comprehensive Parameter Exploration**:
```python
from evolvepro.src.evolve import grid_search

grid_search(
    dataset_name='my_dataset',
    experiment_name='full_sweep',
    model_name='esm2_t48_15B_UR50D',
    embeddings_path='output/plm/esm',
    labels_path='data/dms',
    num_simulations=10,
    num_iterations=[3, 5, 10],  # Multiple values
    measured_var=['activity_scaled', 'activity_binary'],
    learning_strategies=['topn', 'random', 'dist'],
    num_mutants_per_round=[8, 16, 32],
    num_final_round_mutants=10,
    first_round_strategies=['random', 'diverse_medoids'],
    embedding_types=['embeddings', 'embeddings_pca_50'],
    pca_components=[50, 100],
    regression_types=['randomforest', 'ridge', 'gradientboosting'],
    embeddings_file_type='csv',
    output_dir='output/grid_search'
)
```
- **Total combinations**: All combinations of list parameters are tested
- **Output**: Single CSV with all results
- **Use case**: Systematic benchmarking and hyperparameter optimization

---

## Complete Example Workflows

### Example 1: DMS Benchmark

```bash
# Step 1: Process DMS data
python scripts/process/dms_process.py

# Step 2: Extract embeddings (in plm environment)
conda activate plm
python evolvepro/plm/esm/extract.py \
  esm2_t48_15B_UR50D \
  output/dms/brenan.fasta \
  output/plm/esm/brenan \
  --toks_per_batch 512 \
  --include mean \
  --concatenate_dir output/plm/esm/

# Step 3: Run evolution simulation (in evolvepro environment)
conda activate evolvepro
python scripts/dms/dms_main.py \
  --dataset_name brenan \
  --experiment_name standard \
  --model_name esm2_t48_15B_UR50D \
  --embeddings_path output/plm/esm \
  --labels_path data/dms \
  --num_simulations 10 \
  --num_iterations 5 \
  --num_mutants_per_round 16 \
  --num_final_round_mutants 10 \
  --learning_strategies topn \
  --regression_types randomforest \
  --first_round_strategies random \
  --embedding_types embeddings \
  --measured_var activity_scaled \
  --embeddings_file_type csv \
  --output_dir output/dms_results

# Step 4: Visualize results
python scripts/plot/dms.py
```

### Example 2: Experimental Protein Engineering

```bash
# Step 1a: Generate variant library
python scripts/process/exp_process.py

# Step 1b: Or use Python:
python
>>> from evolvepro.src.process import generate_wt, generate_single_aa_mutants
>>> generate_wt('MNTINIAKNDFS...', 'output/exp/my_protein_WT.fasta')
>>> generate_single_aa_mutants('output/exp/my_protein_WT.fasta',
...                             'output/exp/my_protein_library.fasta')

# Step 2: Extract embeddings (in plm environment)
conda activate plm
python evolvepro/plm/esm/extract.py \
  esm2_t48_15B_UR50D \
  output/exp/my_protein_library.fasta \
  output/plm/esm/my_protein \
  --toks_per_batch 512 \
  --include mean \
  --concatenate_dir output/plm/esm/

# Step 3: Predict next round (in evolvepro environment)
conda activate evolvepro
python scripts/exp/my_protein.py

# Step 4: Order top variants for experimental testing
# Review: output/exp_results/my_protein/Round2/df_test.csv
```

---

## Troubleshooting

### Common Issues

**1. GPU Out of Memory**:
- Reduce `--toks_per_batch` parameter
- Use smaller model (e.g., esm2_t36_3B instead of esm2_t48_15B)
- Process sequences in smaller batches

**2. NumPy Version Conflicts**:
- Ensure `numpy<2.0` in evolvepro environment (required by scikit-learn-extra)
- Use separate `plm` environment for embedding extraction

**3. Embeddings-Labels Mismatch**:
- Ensure variant names match exactly between FASTA and labels
- Check for whitespace or case differences
- Verify WT is named consistently ("WT" recommended)

**4. Python Version Issues**:
- PLM environment requires Python 3.10
- Check: `python --version`

**5. Missing Dependencies**:
```bash
# For diverse_medoids strategy
pip install scikit-learn-extra

# For XGBoost
pip install xgboost
```

---

## File Format Reference

### FASTA Format
```
>WT
MNTINIAKNDFSQRWVTLP
>A123G
MNTINIAKNDFSQRWVTLG
>V45I
MNTINIAKNDISQRWVTLP
```

### Labels CSV Format
```csv
variant,activity,activity_scaled,activity_binary
WT,1.0,0.5,0
A123G,1.8,0.85,1
V45I,0.6,0.15,0
```

### Embeddings CSV Format
```csv
,0,1,2,3,...,5119
WT,0.123,-0.456,0.789,...,0.321
A123G,-0.234,0.567,-0.891,...,-0.432
V45I,0.345,-0.678,0.123,...,0.543
```

### Round Data (Excel)
```
| variant | activity | notes          |
|---------|----------|----------------|
| WT      | 1.0      | baseline       |
| A123G   | 1.8      | improved       |
| V45I    | 0.6      | decreased      |
```

---

## Performance Tips

1. **GPU Usage**: Always use GPU for PLM extraction (10-100x faster)
2. **Batch Processing**: Process multiple datasets in parallel
3. **Model Selection**: ESM2-650M or ESM2-3B provide good speed/accuracy tradeoff
4. **Caching**: Reuse embeddings across multiple evolution experiments
5. **Simulation Count**: Use 10-20 simulations for robust statistics
6. **PCA**: Consider PCA for datasets with <100 training sequences

---

## Summary of Key Commands

**Check current environment**:
```bash
conda env list
```

**Activate environments**:
```bash
conda activate evolvepro  # For Steps 1, 3, 4
conda activate plm        # For Step 2 only
```

**Quick DMS run**:
```bash
python scripts/dms/dms_main.py \
  --dataset_name my_data --experiment_name test \
  --model_name esm2_t48_15B_UR50D \
  --embeddings_path output/plm/esm --labels_path data/dms \
  --num_simulations 5 --num_iterations 5 \
  --num_mutants_per_round 16 --num_final_round_mutants 10 \
  --learning_strategies topn --regression_types randomforest \
  --first_round_strategies random --embedding_types embeddings \
  --measured_var activity_scaled --embeddings_file_type csv \
  --output_dir output/
```

**GPU check**:
```python
import torch
print(torch.cuda.is_available())
```

---

For installation instructions, see README.md or CLAUDE.md.

For questions or issues, please open an issue on GitHub.
