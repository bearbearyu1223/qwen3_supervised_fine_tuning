"""
Evaluate language models on MATH dataset using the r1_zero prompt.

Supported models:
    - Qwen/Qwen2.5-Math-1.5B (default, optimized for math)
    - meta-llama/Llama-3.1-8B (general-purpose, larger)
    - Any HuggingFace causal LM

Running:

```
# Qwen model on NVIDIA GPU (using vLLM backend - default)
python scripts/run_math_eval.py \\
    --input-path data/math/test.jsonl \\
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \\
    --output-path outputs/math_eval_results.jsonl

# Qwen model on Mac M-series with MPS (using transformers backend)
python scripts/run_math_eval.py \\
    --input-path data/math/test.jsonl \\
    --model-name-or-path models/qwen2.5-math-1.5b \\
    --output-path outputs/math_eval_results.jsonl \\
    --backend transformers

# Llama 3.1 8B on NVIDIA GPU (vLLM)
python scripts/run_math_eval.py \\
    --model-name-or-path meta-llama/Llama-3.1-8B \\
    --output-path outputs/llama_eval_results.jsonl \\
    --backend vllm

# Llama 3.1 8B on Mac (transformers, limited samples for memory)
python scripts/run_math_eval.py \\
    --model-name-or-path meta-llama/Llama-3.1-8B \\
    --output-path outputs/llama_eval_results.jsonl \\
    --backend transformers \\
    --num-samples 10

# Quick test with limited samples
python scripts/run_math_eval.py \\
    --model-name-or-path models/qwen2.5-math-1.5b \\
    --output-path outputs/math_eval_results.jsonl \\
    --backend transformers \\
    --num-samples 10
```

Note: For gated models like Llama, first authenticate with: huggingface-cli login
"""
import argparse
import logging
import sys

from cs336_alignment.evaluate_math import evaluate_math

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Evaluate model on MATH dataset with r1_zero prompt"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/math/test.jsonl",
        help="Path to MATH test examples (JSONL format)",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (vllm backend only; ignored on Mac with transformers backend)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/math_eval_results.jsonl",
        help="Path to write output results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all samples in the dataset)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend: 'vllm' for NVIDIA GPUs, 'transformers' for CPU/MPS (Mac)",
    )
    args = parser.parse_args()

    # Warn if --num-gpus is used with transformers backend
    if args.backend == "transformers" and args.num_gpus > 1:
        logger.warning(
            "--num-gpus is ignored with transformers backend. "
            "Mac M-series chips have a single integrated GPU accessed via MPS."
        )

    logger.info("Running %s", " ".join(sys.argv))
    evaluate_math(
        model_name_or_path=args.model_name_or_path,
        input_path=args.input_path,
        output_path=args.output_path,
        num_gpus=args.num_gpus,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.num_samples,
        backend=args.backend,
    )
    logger.info("Finished running %s", sys.argv[0])
