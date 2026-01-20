"""
Run Supervised Fine-Tuning (SFT) on MATH dataset.

This script fine-tunes a language model (e.g., Qwen2.5-Math-1.5B) on math
problems using the r1_zero prompt format with <think> and <answer> tags.

The script automatically detects the hardware platform (CUDA, MPS, or CPU)
and optimizes training parameters accordingly when using --auto mode.

Usage Examples:
    # Automatic hardware detection (recommended)
    uv run python scripts/run_sft.py --auto \\
        --model-name-or-path models/qwen2.5-math-1.5b \\
        --train-data-path data/math/train.jsonl \\
        --output-dir outputs/sft_model

    # Quick test with 100 samples and auto-detection
    uv run python scripts/run_sft.py --auto \\
        --model-name-or-path models/qwen2.5-math-1.5b \\
        --num-samples 100

    # Manual configuration (full control)
    uv run python scripts/run_sft.py \\
        --model-name-or-path models/qwen2.5-math-1.5b \\
        --train-data-path data/math/train.jsonl \\
        --output-dir outputs/sft_model \\
        --num-epochs 1 \\
        --batch-size 1 \\
        --gradient-accumulation-steps 8

    # Evaluate the trained model
    uv run python scripts/run_math_eval.py \\
        --model-name-or-path outputs/sft_model/final \\
        --output-path outputs/sft_eval_results.jsonl \\
        --backend transformers

Output Files:
    The script saves the following files to --output-dir:
    - final/: The final trained model and tokenizer
    - checkpoint-{step}/: Intermediate checkpoints (every --save-steps)
    - training_metrics.json: Loss and learning rate history
    - training_curves.png: Visualization of training curves

Hardware Support:
    - CUDA: Uses bf16/fp16 mixed precision, multi-GPU via Accelerate
    - MPS (Mac): Uses fp32 for stability, memory-efficient settings
    - CPU: Fallback with conservative batch sizes

Notes:
    - Use --auto flag for automatic hardware-optimized configuration
    - Gradient accumulation allows training with larger effective batch sizes
"""
import argparse
import logging
import sys

from cs336_alignment.sft import SFTConfig, train_sft

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SFT training.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Supervised Fine-Tuning (SFT) on MATH dataset with r1_zero format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Model arguments
    # -------------------------------------------------------------------------
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/qwen2.5-math-1.5b",
        help="HuggingFace model name or path to local model directory",
    )

    # -------------------------------------------------------------------------
    # Data arguments
    # -------------------------------------------------------------------------
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train-data-path",
        type=str,
        default="data/math/train.jsonl",
        help="Path to training data in JSONL format (each line: {problem, solution})",
    )
    data_group.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization (longer sequences are truncated)",
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: use all samples)",
    )

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size per device (keep small for limited GPU/MPS memory)",
    )
    train_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps (effective batch = batch_size * this)",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW optimizer",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for AdamW optimizer",
    )
    train_group.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps for linear warmup",
    )
    train_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )

    # -------------------------------------------------------------------------
    # Output and logging
    # -------------------------------------------------------------------------
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft_model",
        help="Directory to save trained model, checkpoints, and metrics",
    )
    output_group.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save a checkpoint every N optimizer update steps",
    )
    output_group.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log training metrics every N optimizer update steps",
    )

    # -------------------------------------------------------------------------
    # Device configuration
    # -------------------------------------------------------------------------
    device_group = parser.add_argument_group("Device")
    device_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for training (auto selects cuda > mps > cpu)",
    )
    device_group.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader num_workers (default: auto-detect based on device)",
    )

    # -------------------------------------------------------------------------
    # Auto-configuration
    # -------------------------------------------------------------------------
    auto_group = parser.add_argument_group("Auto-configuration")
    auto_group.add_argument(
        "--auto",
        action="store_true",
        help="Enable automatic hardware detection and optimization. "
             "Overrides --batch-size, --gradient-accumulation-steps, and --num-workers "
             "with hardware-optimized values.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for SFT training.

    Parses arguments, creates training configuration, runs training,
    and logs the output locations for the trained model and metrics.
    """
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    args = parse_args()

    # Log the full command for reproducibility
    logger.info("Command: %s", " ".join(sys.argv))

    # Create training configuration
    if args.auto:
        # Use automatic hardware detection
        logger.info("Using automatic hardware detection (--auto)")
        config = SFTConfig.create_auto_config(
            model_name_or_path=args.model_name_or_path,
            train_data_path=args.train_data_path,
            output_dir=args.output_dir,
            # Override with explicitly provided args
            max_seq_length=args.max_seq_length,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
        )
        logger.info(
            "Auto-detected: batch_size=%d, gradient_accumulation_steps=%d, num_workers=%d",
            config.batch_size,
            config.gradient_accumulation_steps,
            config.num_workers,
        )
    else:
        # Manual configuration
        # Use -1 (auto) for num_workers if not specified
        num_workers = args.num_workers if args.num_workers is not None else -1

        config = SFTConfig(
            model_name_or_path=args.model_name_or_path,
            train_data_path=args.train_data_path,
            max_seq_length=args.max_seq_length,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_grad_norm=args.max_grad_norm,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            device=args.device,
            num_workers=num_workers,
        )

    # Log effective batch size (use resolved values from config or args)
    batch_size = config.batch_size if config.batch_size > 0 else args.batch_size
    grad_accum = config.gradient_accumulation_steps if config.gradient_accumulation_steps > 0 else args.gradient_accumulation_steps
    effective_batch_size = batch_size * grad_accum
    logger.info(
        "Effective batch size: %d (batch_size=%d × gradient_accumulation_steps=%d)",
        effective_batch_size,
        batch_size,
        grad_accum,
    )

    # Run training
    final_model_path, metrics = train_sft(config, num_samples=args.num_samples)

    # Log training summary
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("  Model:   %s", final_model_path)
    logger.info("  Metrics: %s/training_metrics.json", args.output_dir)
    logger.info("  Plot:    %s/training_curves.png", args.output_dir)

    # Log final training stats
    if metrics.losses:
        logger.info("Final loss: %.4f", metrics.losses[-1])

    # Print evaluation command
    logger.info("")
    logger.info("To evaluate the trained model, run:")
    logger.info(
        "  uv run python scripts/run_math_eval.py \\\n"
        "      --model-name-or-path %s \\\n"
        "      --output-path outputs/sft_eval_results.jsonl \\\n",
        final_model_path,
    )


if __name__ == "__main__":
    main()
