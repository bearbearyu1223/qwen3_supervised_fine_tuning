# Qwen3 Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) and evaluation pipeline for Qwen3 models on math reasoning tasks using the MATH dataset.

## Features

- SFT training with automatic hardware detection (CUDA, MPS, CPU)
- Support for Qwen3-4B, Qwen3-1.7B, and other Qwen models
- Evaluation on MATH dataset with r1_zero prompt format
- Training metrics tracking and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/bearbearyu1223/qwen3_supervised_fine_tuning.git
cd qwen3_supervised_fine_tuning

# Install with uv (recommended)
uv sync

# Install dependecies with CUDA support for flash-attn (optional)
uv sync --extra cuda
```

## Quick Start

### 1. Download Model and Data

```bash
# Download Qwen3-1.7B model
uv run python scripts/download_model.py --model-name Qwen/Qwen3-1.7B

# Download MATH dataset (if not already present)
uv run python scripts/download_math.py
```

### 2. Run Zero-Shot Evaluation

Evaluate the base Qwen3 model before fine-tuning to establish a baseline:

```bash
uv run python scripts/run_math_eval.py \
    --model-name-or-path models/qwen3-1.7b \
    --output-path outputs/qwen3_base_eval.jsonl
```

### 3. Run SFT Training and Evaluation on Lambda Cloud

This guide uses a **1x A100 40GB SXM4** instance on [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud).

#### Step 1: Launch Instance and SSH

1. Go to [Lambda Cloud](https://cloud.lambdalabs.com/) and launch a **1x A100 40GB SXM4** instance
2. SSH into your instance:

```bash
ssh ubuntu@<your-instance-ip>
```

#### Step 2: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/bearbearyu1223/qwen3_supervised_fine_tuning.git
cd qwen3_supervised_fine_tuning

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Install dependencies with CUDA support
uv sync --extra cuda
```

#### Step 3: Download Model and Data

```bash
uv run python scripts/download_model.py --model-name Qwen/Qwen3-1.7B
uv run python scripts/download_math.py
```

#### Step 4: Run SFT Training

```bash
# Run with AUTO mode (auto-detects GPU and optimal settings)
uv run accelerate launch scripts/run_sft.py --auto \
    --model-name-or-path models/qwen3-1.7b \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/sft_qwen3
```

#### Step 5: Evaluate the Trained Model

```bash
uv run python scripts/run_math_eval.py \
    --model-name-or-path outputs/sft_qwen3/final \
    --output-path outputs/sft_qwen3_eval.jsonl
```

## Project Structure

```text
qwen3_supervised_fine_tuning/
├── cs336_alignment/          # Core module
│   ├── sft.py               # SFT training implementation
│   ├── evaluate_math.py     # Evaluation utilities
│   ├── drgrpo_grader.py     # Reward/grading functions
│   └── prompts/             # Prompt templates
├── scripts/                  # CLI scripts
│   ├── run_sft.py           # SFT training script
│   ├── run_math_eval.py     # Evaluation script
│   ├── download_model.py    # Model download utility
│   └── download_math.py     # Data download utility
├── data/math/               # MATH dataset
└── pyproject.toml           # Project configuration
```

## License

This project is for educational purposes.
