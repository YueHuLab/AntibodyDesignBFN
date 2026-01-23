# AntibodyDesignBFN
**AntibodyDesignBFN** is a robust framework for fixed-backbone antibody sequence design based on **Bayesian Flow Networks (BFN)**. It leverages **Geometric Transformers** with **Invariant Point Attention (IPA)** to model the joint distribution of sequence and structure (or sequence conditioned on structure) in a continuous-time generative process.
## Features
- **Fixed-Backbone Design**: Efficiently design CDR sequences for a given antibody backbone structure.
- **Bayesian Flow Networks**: accurate, iterative refinement of discrete sequences.
- **Geometric Transformer**: E(3)-invariant architecture capturing precise side-chain packing and geometric constraints.
- **Low Compute**: Optimized for Apple Silicon (tested on **Mac Mini M4**).
## Installation
### Prerequisites
- Python 3.8+
- PyTorch (compatible with your CUDA version)
### Steps
1. Clone the repository (if not already downloaded).
2. Install the package in editable mode:
```bash
cd AntibodyDesignBFN
pip install -e .
```
This will install `antibody_bfn` and necessary dependencies (`torch`, `biopython`, `lmdb`, etc.).
## Model Checkpoints and test dataset 
Pre-trained model checkpoints are available on Hugging Face:
[https://huggingface.co/YueHuLab/AntibodyDesignBFN/](https://huggingface.co/YueHuLab/AntibodyDesignBFN/)
Please download the checkpoint `.pt` file and update your configuration file (e.g., `configs/demo_design.yml`) to point to the local path.
## Training Data
The model is trained on the **Structural Antibody Database (SAbDab)**.
- **Summary File**: `./data/sabdab_summary_all.tsv`
- **Structure Directory**: `./data/all_structures/chothia`
- **Processing**: The model automatically preprocesses PDB files into an LMDB cache for efficient training.
## Usage
### 1. Sequence Design (Inference)
To design sequences for a target antibody structure (PDB file), use `design_seq.py`.
**Example:**
```bash
python design_seq.py ../data/examples/7DK2_AB_C.pdb \
    --heavy A \
    --light B \
    --config configs/demo_design.yml \
    --device cuda
```
**Arguments:**
- `pdb_path`: Path to the input PDB file.
- `--heavy`: Chain ID of the heavy chain (default: 'H').
- `--light`: Chain ID of the light chain (default: 'L').
- `--config`: Path to the configuration file (e.g., `configs/demo_design.yml`).
- `--num_samples`: Number of sequences to generate (default: 1).
- `--stochastic`: Enable stochastic sampling (diverse outputs).
- `--eval`: Evaluation mode (compares generated sequence to the native sequence in the PDB).
**Configuration:**
Ensure your config file points to a valid model checkpoint:
```yaml
model:
  checkpoint: /path/to/your/checkpoint.pt
```
### 2. Training
To train a new model from scratch or resume training:
```bash
python train.py configs/train/bfn_seq_design.yml
```
**Resuming Training:**
```bash
python train.py configs/train/bfn_seq_design.yml --resume /path/to/checkpoint.pt
```
## Directory Structure
- `antibody_bfn/`: Core package source code.
  - `models/`: BFN and Geometric Transformer architectures.
  - `modules/`: Neural network layers (IPA, attention).
  - `datasets/`: Data loading logic (SAbDab, custom).
- `configs/`: YAML configuration files for training and testing.
- `design_seq.py`: Evaluation and inference script.
- `train.py`: Training script.
#test dataset
python batch_evaluate_checkpoints.py \
  --config configs/test/bfn_testset.yml \
  --test_set data/2025_testset_43.csv \
  --chothia_dir data/2025_pdbs \
  --device mps \
  --ckpt_dir ./logs/bfn_seq_design_finetune_2026_01_23__16_24_38/checkpoints \
  --start_ckpt 2400 \
  --end_ckpt 4100 \
  --step 25 \
  --output_base_dir ./results/batch_evaluation_new


  python evaluate_testset.py \
  --config configs/test/bfn_testset.yml \
  --test_set data/2025_testset_43.csv \
  --chothia_dir data/2025_pdbs \
  --device mps
## License
MIT License
