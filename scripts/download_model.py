"""
Download models from HuggingFace for local use.

Supported models:
    - Qwen/Qwen2.5-Math-1.5B (1.5B params, ~3GB) - default, optimized for math
    - Qwen/Qwen3-4B (4B params, ~8GB) - latest Qwen3 model
    - Qwen/Qwen3-1.7B (1.7B params, ~4GB) - smaller Qwen3
    - Qwen/Qwen2.5-0.5B (0.5B params, ~1GB) - for testing
    - meta-llama/Llama-3.1-8B (8B params, ~16GB) - requires HF token

Running:

```
# Download Qwen 2.5 Math 1.5B (default, saves to models/qwen2.5-math-1.5b)
python scripts/download_model.py

# Download Llama 3.1 8B (requires HuggingFace login)
# First: huggingface-cli login
python scripts/download_model.py --model-name meta-llama/Llama-3.1-8B

# Download a specific model to custom location
python scripts/download_model.py --model-name Qwen/Qwen2.5-Math-1.5B --output-dir models/my-model
```

By default, models are saved to models/<model-name-lowercase>.
For example: Qwen/Qwen2.5-Math-1.5B -> models/qwen2.5-math-1.5b

Note: For gated models like Llama, you need to:
1. Accept the license on HuggingFace: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Login with: huggingface-cli login
"""
import argparse
import logging
import os

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-Math-1.5B"

# Known models and their approximate sizes
KNOWN_MODELS = {
    # Qwen models
    "Qwen/Qwen2.5-Math-1.5B": {"params": "1.5B", "size_gb": 3, "gated": False},
    "Qwen/Qwen2.5-0.5B": {"params": "0.5B", "size_gb": 1, "gated": False},
    "Qwen/Qwen3-4B": {"params": "4B", "size_gb": 8, "gated": False},
    "Qwen/Qwen3-1.7B": {"params": "1.7B", "size_gb": 4, "gated": False},
    "Qwen/Qwen3-0.6B": {"params": "0.6B", "size_gb": 1.5, "gated": False},
    # Llama models
    "meta-llama/Llama-3.1-8B": {"params": "8B", "size_gb": 16, "gated": True},
    "meta-llama/Llama-3.2-1B": {"params": "1B", "size_gb": 2, "gated": True},
    "meta-llama/Llama-3.2-3B": {"params": "3B", "size_gb": 6, "gated": True},
}


def main(model_name: str, output_dir: str | None = None, token: str | None = None):
    """
    Download a model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.1-8B")
        output_dir: Directory to save the model. If None, uses models/<model-name>
        token: HuggingFace token for gated models. If None, uses cached login.
    """
    if output_dir is None:
        # Default to models/<model-name> with simple lowercase naming
        # e.g., Qwen/Qwen2.5-Math-1.5B -> models/qwen2.5-math-1.5b
        simple_name = model_name.split("/")[-1].lower()
        output_dir = os.path.join("models", simple_name)

    os.makedirs(output_dir, exist_ok=True)

    # Check if this is a known gated model
    model_info = KNOWN_MODELS.get(model_name, {})
    is_gated = model_info.get("gated", False)

    if is_gated:
        logger.info(f"Note: {model_name} is a gated model requiring HuggingFace authentication.")
        logger.info("If download fails, ensure you have:")
        logger.info("  1. Accepted the license at: https://huggingface.co/%s", model_name)
        logger.info("  2. Logged in with: huggingface-cli login")

    if model_info:
        logger.info(f"Model info: {model_info.get('params')} parameters, ~{model_info.get('size_gb')}GB download")

    logger.info(f"Downloading model {model_name} from HuggingFace...")
    logger.info(f"Saving to {output_dir}")

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token,  # Use provided token or cached login
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            logger.error("Authentication failed for gated model.")
            logger.error("Please ensure you have:")
            logger.error("  1. Accepted the license at: https://huggingface.co/%s", model_name)
            logger.error("  2. Logged in with: huggingface-cli login")
            raise
        raise

    logger.info(f"Model saved to {output_dir}")
    logger.info(f"You can now use this model with: --model-name-or-path {output_dir}")


def list_known_models():
    """Print a list of known supported models."""
    print("Known supported models:")
    print("-" * 70)
    for name, info in KNOWN_MODELS.items():
        gated_str = " [GATED - requires HF login]" if info.get("gated") else ""
        print(f"  {name}")
        print(f"    Parameters: {info['params']}, Size: ~{info['size_gb']}GB{gated_str}")
    print("-" * 70)
    print("\nFor gated models, first run: huggingface-cli login")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Download a model from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default Qwen model
  python scripts/download_model.py

  # Download Llama 3.1 8B (requires HF login first)
  huggingface-cli login
  python scripts/download_model.py --model-name meta-llama/Llama-3.1-8B

  # List known models
  python scripts/download_model.py --list
        """,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the model (default: models/<model-name>)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or use 'huggingface-cli login')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known supported models and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_known_models()
    else:
        main(args.model_name, args.output_dir, args.token)
