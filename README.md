# Qwen3 Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) and evaluation pipeline for Qwen3 models on math reasoning tasks using the MATH dataset.

## Features

- SFT training with automatic hardware detection (CUDA, MPS, CPU)
- Support for Qwen3-4B, Qwen3-1.7B, and other Qwen models
- Evaluation on MATH dataset with r1_zero prompt format
- Training metrics tracking and visualization
- Colab notebook for cloud training

## Installation

```bash
# Clone the repository
git clone https://github.com/bearbearyu1223/qwen3_supervised_fine_tuning.git
cd qwen3_supervised_fine_tuning

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# For CUDA support (NVIDIA GPUs)
uv sync --extra cuda
```

## Quick Start

### 1. Download Model and Data

```bash
# Download Qwen3-4B model
uv run python scripts/download_model.py --model Qwen/Qwen3-4B --output models/qwen3-4b

# Download MATH dataset (if not already present)
uv run python scripts/download_math.py
```

### 2. Run SFT Training

```bash
# With automatic hardware detection
uv run accelerate launch scripts/run_sft.py --auto \
    --model-name-or-path models/qwen3-4b \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/sft_qwen3

# With custom settings
uv run accelerate launch scripts/run_sft.py \
    --model-name-or-path models/qwen3-4b \
    --train-data-path data/math/train.jsonl \
    --output-dir outputs/sft_qwen3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --max-seq-length 512
```

### 3. Evaluate Model

```bash
# Evaluate fine-tuned model
uv run python scripts/run_math_eval.py \
    --model outputs/sft_qwen3/final \
    --input data/math/test.jsonl \
    --output outputs/eval_results.jsonl
```

## Project Structure

```
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
├── notebooks/               # Colab notebooks
└── pyproject.toml           # Project configuration
```

## Memory Optimization

For large models like Qwen3-4B on limited GPU memory:

1. **Reduce sequence length**: `--max-seq-length 512`
2. **Use DeepSpeed ZeRO**: Create a `ds_config.json` with CPU offloading
3. **Enable gradient checkpointing**: Add `model.gradient_checkpointing_enable()` after model loading

## Supported Models

| Model | Parameters | Min VRAM |
|-------|------------|----------|
| Qwen/Qwen3-4B | 4B | ~24GB |
| Qwen/Qwen3-1.7B | 1.7B | ~8GB |
| Qwen/Qwen2.5-Math-1.5B | 1.5B | ~6GB |

## License

This project is for educational purposes.
