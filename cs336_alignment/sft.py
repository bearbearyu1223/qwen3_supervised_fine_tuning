"""
Supervised Fine-Tuning (SFT) for math reasoning models.

This module provides utilities for training language models on the MATH dataset
using the r1_zero prompt format. Supports multiple model architectures and
automatic hardware detection for optimal training configuration.

Supported models:
    - Qwen/Qwen2.5-Math-1.5B (1.5B params, ~3GB VRAM) - optimized for math
    - Qwen/Qwen3-4B (4B params, ~8GB VRAM) - latest Qwen3 model
    - Qwen/Qwen3-1.7B (1.7B params, ~4GB VRAM) - smaller Qwen3
    - Qwen/Qwen2.5-0.5B (0.5B params, ~1GB VRAM) - for testing
    - meta-llama/Llama-3.1-8B (8B params, ~16GB VRAM)
    - meta-llama/Llama-3.2-1B (1B params, ~2GB VRAM)
    - Any HuggingFace causal LM

Key components:
    - detect_compute_environment: Auto-detect hardware and recommend settings
    - SFTConfig: Training configuration with auto-detection support
    - train_sft: Main training loop with metrics tracking
    - TrainingMetrics: Container for tracking and plotting training metrics

Example usage:
    >>> from cs336_alignment.sft import SFTConfig, train_sft
    >>> # Auto-detect optimal settings
    >>> config = SFTConfig.create_auto_config(
    ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
    ...     train_data_path="data/math/train.jsonl",
    ... )
    >>> model_path, metrics = train_sft(config, num_samples=100)

    >>> # Train Llama model (requires more memory)
    >>> config = SFTConfig.create_auto_config(
    ...     model_name_or_path="meta-llama/Llama-3.1-8B",
    ... )
    >>> model_path, metrics = train_sft(config)
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from xopen import xopen

logger = logging.getLogger(__name__)

# Load the r1_zero prompt template
PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"
with open(PROMPT_PATH) as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read()


@dataclass
class SFTConfig:
    """
    Configuration for SFT training with automatic hardware optimization.

    This config supports automatic detection and optimization of training
    parameters based on the available hardware. Use "auto" values to let
    the system choose optimal settings, or specify values explicitly.

    Attributes:
        model_name_or_path: HuggingFace model ID or local path.
        train_data_path: Path to training data in JSONL format.
        max_seq_length: Maximum sequence length for tokenization.
        num_epochs: Number of training epochs.
        batch_size: Batch size per device. Use -1 for auto-detection based
            on available memory.
        gradient_accumulation_steps: Steps to accumulate gradients. Use -1
            for auto-detection to achieve effective batch size of ~8.
        learning_rate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for regularization.
        warmup_ratio: Fraction of training steps for learning rate warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        output_dir: Directory to save model checkpoints and metrics.
        save_steps: Save checkpoint every N optimizer steps.
        logging_steps: Log metrics every N optimizer steps.
        device: Device for training ("auto", "cuda", "mps", or "cpu").
        num_workers: DataLoader num_workers. Use -1 for auto-detection.

    Example:
        >>> # Fully automatic configuration
        >>> config = SFTConfig.create_auto_config(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     train_data_path="data/math/train.jsonl",
        ... )
        >>> print(f"Using batch_size={config.batch_size}")

        >>> # Manual configuration
        >>> config = SFTConfig(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     batch_size=2,
        ...     gradient_accumulation_steps=4,
        ... )
    """
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B"

    # Data
    train_data_path: str = "data/math/train.jsonl"
    max_seq_length: int = 1024

    # Training
    num_epochs: int = 1
    batch_size: int = -1  # -1 = auto-detect based on hardware
    gradient_accumulation_steps: int = -1  # -1 = auto-detect
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "outputs/sft_model"
    save_steps: int = 500
    logging_steps: int = 10

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    num_workers: int = -1  # -1 = auto-detect

    @classmethod
    def create_auto_config(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-Math-1.5B",
        train_data_path: str = "data/math/train.jsonl",
        output_dir: str = "outputs/sft_model",
        **kwargs,
    ) -> "SFTConfig":
        """
        Create a config with auto-detected optimal settings for the hardware.

        Detects the available compute environment and sets batch_size,
        gradient_accumulation_steps, and num_workers automatically based on
        both hardware capabilities and estimated model size.

        Args:
            model_name_or_path: HuggingFace model ID or local path. The model size
                is estimated from the name to provide appropriate memory settings.
            train_data_path: Path to training data.
            output_dir: Directory for outputs.
            **kwargs: Additional config overrides.

        Returns:
            SFTConfig with hardware-optimized settings.
        """
        # Pass model name for size-aware recommendations
        env = detect_compute_environment(model_name_or_path)

        # Log detected environment and model size
        model_size = estimate_model_size_billions(model_name_or_path)
        logger.info(f"Model size estimate: {model_size}B parameters")
        logger.info(f"Detected compute environment: {env}")

        return cls(
            model_name_or_path=model_name_or_path,
            train_data_path=train_data_path,
            output_dir=output_dir,
            batch_size=kwargs.pop("batch_size", env.recommended_batch_size),
            gradient_accumulation_steps=kwargs.pop(
                "gradient_accumulation_steps", env.recommended_grad_accum
            ),
            num_workers=kwargs.pop("num_workers", env.recommended_num_workers),
            device=kwargs.pop("device", env.device),
            **kwargs,
        )


def get_device(device_str: str = "auto") -> str:
    """
    Get the best available device for training.

    Checks device availability in order of preference: CUDA > MPS > CPU.

    Args:
        device_str: Device specification. Use "auto" for automatic detection,
            or specify "cuda", "mps", or "cpu" explicitly.

    Returns:
        Device identifier string ("cuda", "mps", or "cpu").
    """
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class ComputeEnvironment:
    """
    Detected compute environment information.

    This dataclass contains details about the available hardware and
    recommended training configurations based on the detected platform.

    Attributes:
        device: Primary device type ("cuda", "mps", or "cpu").
        device_name: Human-readable device name.
        num_gpus: Number of available GPUs (0 for CPU/MPS).
        total_memory_gb: Total device memory in GB (VRAM for GPU, RAM for CPU).
        supports_bf16: Whether the device supports bfloat16.
        supports_fp16: Whether the device supports float16.
        recommended_dtype: Recommended torch dtype for this device.
        recommended_batch_size: Suggested batch size for typical 1.5B model.
        recommended_grad_accum: Suggested gradient accumulation steps.
        recommended_num_workers: Suggested DataLoader num_workers.
        mixed_precision: Recommended mixed precision setting for Accelerator.
    """
    device: str
    device_name: str
    num_gpus: int
    total_memory_gb: float
    supports_bf16: bool
    supports_fp16: bool
    recommended_dtype: torch.dtype
    recommended_batch_size: int
    recommended_grad_accum: int
    recommended_num_workers: int
    mixed_precision: str

    def __str__(self) -> str:
        return (
            f"ComputeEnvironment(\n"
            f"  device={self.device}, name='{self.device_name}',\n"
            f"  num_gpus={self.num_gpus}, memory={self.total_memory_gb:.1f}GB,\n"
            f"  bf16={self.supports_bf16}, fp16={self.supports_fp16},\n"
            f"  recommended: dtype={self.recommended_dtype}, "
            f"batch_size={self.recommended_batch_size}, "
            f"grad_accum={self.recommended_grad_accum}\n"
            f")"
        )


# Known model sizes for memory estimation
KNOWN_MODEL_SIZES = {
    # Qwen 2.5 models
    "qwen2.5-math-1.5b": 1.5,
    "qwen2.5-0.5b": 0.5,
    "qwen2.5-1.5b": 1.5,
    "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0,
    # Qwen 3 models
    "qwen3-0.6b": 0.6,
    "qwen3-1.7b": 1.7,
    "qwen3-4b": 4.0,
    "qwen3-8b": 8.0,
    "qwen3-14b": 14.0,
    "qwen3-32b": 32.0,
    # Llama models
    "llama-3.1-8b": 8.0,
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "llama-3.1-70b": 70.0,
}


def estimate_model_size_billions(model_name_or_path: str) -> float:
    """
    Estimate the number of parameters (in billions) for a model.

    Uses known model sizes or parses the model name to estimate size.

    Args:
        model_name_or_path: HuggingFace model ID or local path.

    Returns:
        Estimated number of parameters in billions. Defaults to 1.5B if unknown.

    Example:
        >>> estimate_model_size_billions("meta-llama/Llama-3.1-8B")
        8.0
        >>> estimate_model_size_billions("Qwen/Qwen2.5-Math-1.5B")
        1.5
    """
    import re

    # Normalize the model name
    model_lower = model_name_or_path.lower()
    model_basename = model_lower.split("/")[-1]

    # Check against known models
    for known_name, size in KNOWN_MODEL_SIZES.items():
        if known_name in model_basename:
            return size

    # Try to parse size from name (e.g., "1.5b", "7b", "70b")
    # Match patterns like "1.5b", "7b", "70b", "1b"
    match = re.search(r"(\d+\.?\d*)b", model_basename)
    if match:
        return float(match.group(1))

    # Default to 1.5B if unknown
    logger.warning(f"Unknown model size for {model_name_or_path}, assuming 1.5B parameters")
    return 1.5


def get_recommended_batch_size_for_model(
    model_size_billions: float,
    available_memory_gb: float,
    device: str,
) -> tuple[int, int]:
    """
    Get recommended batch size and gradient accumulation for a model.

    Estimates memory requirements based on model size and available hardware,
    then recommends batch_size and gradient_accumulation_steps to achieve
    a reasonable effective batch size.

    Args:
        model_size_billions: Model size in billions of parameters.
        available_memory_gb: Available device memory in GB.
        device: Device type ("cuda", "mps", or "cpu").

    Returns:
        Tuple of (batch_size, gradient_accumulation_steps).
    """
    # Rough memory estimation (bytes per parameter):
    # - fp32: 4 bytes per param
    # - fp16/bf16: 2 bytes per param
    # - Plus optimizer states, activations, gradients (~4x model size for training)
    # Rule of thumb: need ~6x model size in GB for fp16 training with batch_size=1

    if device == "cuda":
        # CUDA: can use fp16, need ~6x model size in GB
        memory_per_batch = model_size_billions * 6  # Approximate GB per batch item
    elif device == "mps":
        # MPS: shared memory, use fp32, more conservative
        memory_per_batch = model_size_billions * 12  # fp32 + shared memory overhead
    else:
        # CPU: very conservative
        memory_per_batch = model_size_billions * 16

    # Calculate max batch size that fits in memory (with safety margin)
    safety_margin = 0.7  # Use only 70% of available memory
    max_batch_size = max(1, int((available_memory_gb * safety_margin) / memory_per_batch))

    # Cap batch size based on device
    if device == "cuda":
        max_batch_size = min(max_batch_size, 8)  # Cap at 8 for CUDA
    elif device == "mps":
        max_batch_size = min(max_batch_size, 2)  # Cap at 2 for MPS
    else:
        max_batch_size = 1  # CPU: always batch_size=1

    # Calculate gradient accumulation to achieve effective batch size of ~8
    target_effective_batch = 8
    grad_accum = max(1, target_effective_batch // max_batch_size)

    return max_batch_size, grad_accum


def detect_compute_environment(model_name_or_path: str | None = None) -> ComputeEnvironment:
    """
    Detect the available compute environment and recommend optimal settings.

    Automatically detects the hardware platform (CUDA GPUs, Apple MPS, or CPU)
    and returns recommended training configurations based on the detected
    capabilities and memory. If a model name is provided, adjusts recommendations
    based on the model's estimated size.

    Args:
        model_name_or_path: Optional model identifier for size-aware recommendations.
            If provided, batch size recommendations will account for model size.
            Supported models: Qwen, Llama, or any model with size in name (e.g., "7b").

    Returns:
        ComputeEnvironment dataclass with detected capabilities and recommendations.

    Example:
        >>> # Default recommendations (assumes 1.5B model)
        >>> env = detect_compute_environment()
        >>> print(f"Using {env.device_name} with {env.total_memory_gb:.1f}GB memory")

        >>> # Size-aware recommendations for Llama 8B
        >>> env = detect_compute_environment("meta-llama/Llama-3.1-8B")
        >>> print(f"Recommended batch size: {env.recommended_batch_size}")
    """
    import os

    # Estimate model size if provided
    model_size_b = estimate_model_size_billions(model_name_or_path) if model_name_or_path else 1.5

    if torch.cuda.is_available():
        # CUDA GPU detected
        num_gpus = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Check precision support
        supports_bf16 = torch.cuda.is_bf16_supported()
        supports_fp16 = True  # All CUDA GPUs support fp16

        # Determine recommended dtype and mixed precision
        if supports_bf16:
            recommended_dtype = torch.bfloat16
            mixed_precision = "bf16"
        else:
            recommended_dtype = torch.float16
            mixed_precision = "fp16"

        # Recommend batch size based on VRAM and model size
        recommended_batch_size, recommended_grad_accum = get_recommended_batch_size_for_model(
            model_size_billions=model_size_b,
            available_memory_gb=total_memory_gb,
            device="cuda",
        )

        # Use multiple workers for CUDA
        recommended_num_workers = min(4, os.cpu_count() or 1)

        return ComputeEnvironment(
            device="cuda",
            device_name=device_name,
            num_gpus=num_gpus,
            total_memory_gb=total_memory_gb,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            recommended_dtype=recommended_dtype,
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_num_workers=recommended_num_workers,
            mixed_precision=mixed_precision,
        )

    elif torch.backends.mps.is_available():
        # Apple Silicon MPS detected
        import subprocess

        # Try to get Mac model and memory info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            total_memory_gb = int(result.stdout.strip()) / (1024**3)
        except Exception:
            total_memory_gb = 8.0  # Default assumption

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            cpu_brand = result.stdout.strip()
            if "Apple" in cpu_brand:
                device_name = cpu_brand
            else:
                device_name = "Apple Silicon (MPS)"
        except Exception:
            device_name = "Apple Silicon (MPS)"

        # MPS limitations: bf16 has limited support, fp16 works but fp32 is most stable
        supports_bf16 = False  # MPS bf16 support is limited
        supports_fp16 = True

        # MPS works best with fp32 for stability
        recommended_dtype = torch.float32
        mixed_precision = "no"  # fp32 for MPS stability

        # Recommend batch size based on memory and model size
        recommended_batch_size, recommended_grad_accum = get_recommended_batch_size_for_model(
            model_size_billions=model_size_b,
            available_memory_gb=total_memory_gb,
            device="mps",
        )

        # MPS doesn't benefit much from DataLoader workers
        recommended_num_workers = 0

        return ComputeEnvironment(
            device="mps",
            device_name=device_name,
            num_gpus=0,
            total_memory_gb=total_memory_gb,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            recommended_dtype=recommended_dtype,
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_num_workers=recommended_num_workers,
            mixed_precision=mixed_precision,
        )

    else:
        # CPU fallback
        import psutil

        try:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            total_memory_gb = 8.0

        cpu_count = os.cpu_count() or 1
        device_name = f"CPU ({cpu_count} cores)"

        # CPU: use fp32 for stability
        supports_bf16 = False
        supports_fp16 = False
        recommended_dtype = torch.float32
        mixed_precision = "no"

        # Recommend batch size based on memory and model size
        recommended_batch_size, recommended_grad_accum = get_recommended_batch_size_for_model(
            model_size_billions=model_size_b,
            available_memory_gb=total_memory_gb,
            device="cpu",
        )
        recommended_num_workers = min(2, cpu_count)

        return ComputeEnvironment(
            device="cpu",
            device_name=device_name,
            num_gpus=0,
            total_memory_gb=total_memory_gb,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            recommended_dtype=recommended_dtype,
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_num_workers=recommended_num_workers,
            mixed_precision=mixed_precision,
        )


# ==============================================================================
# Core Helper Functions
# ==============================================================================


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size, "prompt_strs and output_strs must have same length"

    all_input_ids = []
    all_labels = []
    all_response_masks = []
    prompt_and_output_lens = []

    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately (no special tokens)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # Concatenate prompt and output
        full_ids = prompt_ids + output_ids
        prompt_and_output_lens.append(len(full_ids))

        # Create input_ids and labels (shifted by 1 for next token prediction)
        # input_ids: all tokens except the last
        # labels: all tokens except the first
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Create response mask: 1 for response tokens in labels, 0 for prompt tokens
        # Since labels are shifted, the response starts at position (prompt_len - 1)
        response_mask = [0] * len(labels)
        response_start = max(0, len(prompt_ids) - 1)
        for i in range(response_start, len(labels)):
            response_mask[i] = 1

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_response_masks.append(response_mask)

    # Pad to max length (max(prompt_and_output_lens) - 1)
    max_len = max(prompt_and_output_lens) - 1
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_labels = []
    padded_response_masks = []

    for input_ids, labels, response_mask in zip(all_input_ids, all_labels, all_response_masks):
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 is ignored in loss
            response_mask = response_mask + [0] * pad_len

        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        padded_response_masks.append(response_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
        "response_mask": torch.tensor(padded_response_masks, dtype=torch.float),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Uses numerically stable computation via log_softmax.

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length). The entropy for each
        next-token prediction.
    """
    # Use log_softmax for numerical stability
    # log_softmax(x) = x - logsumexp(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)  # (batch_size, seq_length, vocab_size)

    # Convert to probabilities for entropy calculation
    probs = torch.exp(log_probs)

    # Entropy: H(p) = -sum(p(x) * log(p(x)))
    # Using the numerically stable form: -sum(probs * log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_length)

    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from
    a causal language model, and optionally the entropy of the model's next-token
    distribution.

    Args:
        model: PreTrainedModel, HuggingFace model used for scoring (placed on the
            correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor of shape (batch_size, sequence_length), concatenated
            prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor of shape (batch_size, sequence_length), labels as
            produced by your tokenization method.
        return_token_entropy: bool, If True, also return per-token entropy by
            calling compute_entropy.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": shape (batch_size, sequence_length), conditional log-probabilities
                log p_θ(x_t | x_{<t}).
            "token_entropy": optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if return_token_entropy=True).
    """
    # Get logits from model
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (batch_size, seq_length, vocab_size)

    # Compute log probabilities using log_softmax (numerically stable)
    log_probs_all = F.log_softmax(logits.float(), dim=-1)  # (batch_size, seq_length, vocab_size)

    # Gather log probs for the target labels
    # We need log_probs[batch, position, label[batch, position]]
    # Use gather to select the log prob of each label
    # labels shape: (batch_size, seq_length)
    # We need to handle -100 labels (padding) - temporarily replace with 0, then mask
    labels_for_gather = labels.clone()
    labels_for_gather[labels == -100] = 0

    # Gather: select log_probs at the label indices
    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels_for_gather.unsqueeze(-1)
    ).squeeze(-1)  # (batch_size, seq_length)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float = 1.0,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those
    elements where mask == 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included
            in the sum.
        normalize_constant: float, the constant to divide by for normalization.
        dim: int | None, the dimension to sum along before normalization. If None,
            sum over all dimensions.

    Returns:
        torch.Tensor, the normalized sum, where masked elements (mask == 0)
        don't contribute to the sum.
    """
    # Apply mask to tensor
    masked_tensor = tensor * mask.float()

    # Sum along dimension (or all dimensions if dim is None)
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)

    # Normalize by constant
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT.

    The SFT loss is the negative log-likelihood of the target output given the prompt.
    We sum over response tokens (masked) and normalize by a constant.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length),
            per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length),
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, number of microbatches per optimizer step.
        normalize_constant: float, the constant by which to divide the sum.
            It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor. The microbatch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata: dict with metadata from the underlying loss call, and any
                other statistics you might want to log.
    """
    batch_size = policy_log_probs.shape[0]

    # SFT loss is negative log likelihood: -sum(log_probs) / normalize_constant
    # We use masked_normalize to sum only over response tokens
    nll_loss = masked_normalize(
        tensor=-policy_log_probs,  # Negative log probs
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None,  # Sum over all dimensions
    )

    # Scale loss for gradient accumulation and batch size (for proper gradient averaging)
    # This ensures gradients are averaged across the effective batch
    scaled_loss = nll_loss / (batch_size * gradient_accumulation_steps)

    # Backward pass
    scaled_loss.backward()

    # Compute metadata
    num_response_tokens = response_mask.sum().item()
    metadata = {
        "num_response_tokens": torch.tensor(num_response_tokens),
        "unscaled_loss": nll_loss.detach(),
    }

    # Return the scaled loss (adjusted for gradient accumulation) for logging
    return scaled_loss.detach(), metadata


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truth_answers: list[str],
    reward_fn: Callable | None = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    device: str = "auto",
) -> dict[str, Any]:
    """
    Log generations from the model for given prompts.

    Args:
        model: The model to generate from.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        ground_truth_answers: List of ground truth answers.
        reward_fn: Optional function to compute rewards. Should take (response, ground_truth)
            and return a dict with "reward", "format_reward", "answer_reward".
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        device: Device to use for generation.

    Returns:
        dict containing:
            - "prompts": list of prompts
            - "generations": list of generated responses
            - "ground_truths": list of ground truth answers
            - "rewards": list of reward dicts (if reward_fn provided)
            - "avg_entropy": average token entropy
            - "avg_response_length": average response length
            - "avg_correct_length": average length for correct responses
            - "avg_incorrect_length": average length for incorrect responses
    """
    if device == "auto":
        device = get_device()

    model.eval()
    results = {
        "prompts": prompts,
        "generations": [],
        "ground_truths": ground_truth_answers,
        "rewards": [],
        "entropies": [],
        "response_lengths": [],
    }

    with torch.no_grad():
        for prompt, gt_answer in zip(prompts, ground_truth_answers):
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Decode generated tokens (excluding prompt)
            generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results["generations"].append(generated_text)
            results["response_lengths"].append(len(generated_ids))

            # Compute entropy if scores are available
            if outputs.scores:
                logits = torch.stack(outputs.scores, dim=1)  # (1, gen_len, vocab_size)
                entropy = compute_entropy(logits)  # (1, gen_len)
                avg_entropy = entropy.mean().item()
                results["entropies"].append(avg_entropy)

            # Compute reward if function provided
            if reward_fn is not None:
                reward_info = reward_fn(generated_text, gt_answer)
                results["rewards"].append(reward_info)

    # Compute summary statistics
    results["avg_entropy"] = sum(results["entropies"]) / len(results["entropies"]) if results["entropies"] else 0.0
    results["avg_response_length"] = sum(results["response_lengths"]) / len(results["response_lengths"]) if results["response_lengths"] else 0.0

    # Compute correct/incorrect lengths if rewards available
    if results["rewards"]:
        correct_lengths = [
            length for length, reward in zip(results["response_lengths"], results["rewards"])
            if reward.get("answer_reward", 0) > 0
        ]
        incorrect_lengths = [
            length for length, reward in zip(results["response_lengths"], results["rewards"])
            if reward.get("answer_reward", 0) == 0
        ]
        results["avg_correct_length"] = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0
        results["avg_incorrect_length"] = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0

    model.train()
    return results


# ==============================================================================
# Dataset and Data Loading
# ==============================================================================


def format_math_example_for_sft(problem: str, solution: str, answer: str) -> tuple[str, str]:
    """
    Format a MATH example for SFT training using r1_zero format.

    The r1_zero format expects:
    - Prompt: "A conversation between... User: {question}\nAssistant: <think>"
    - Response: "{reasoning}</think> <answer>{answer}</answer>"

    Args:
        problem: The math problem
        solution: The step-by-step solution
        answer: The final answer

    Returns:
        Tuple of (prompt, response) strings
    """
    # Format the prompt using r1_zero template
    prompt = R1_ZERO_PROMPT_TEMPLATE.format(question=problem)

    # Format the response: solution as thinking, answer in answer tags
    # Clean up the solution - remove the final answer if it's already in boxed format
    reasoning = solution.strip()

    # Format response with thinking and answer tags
    response = f"{reasoning}</think> <answer>\\boxed{{{answer}}}</answer>"

    return prompt, response


class MathSFTDataset(Dataset):
    """Dataset for SFT on MATH data with r1_zero format."""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 1024,
        num_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []

        # Load and process data
        logger.info(f"Loading data from {data_path}...")
        with xopen(data_path) as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                example = json.loads(line)
                self.examples.append(example)

        logger.info(f"Loaded {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format for SFT
        prompt, response = format_math_example_for_sft(
            problem=example["problem"],
            solution=example["solution"],
            answer=example["answer"],
        )

        # Tokenize prompt and response separately to create response mask
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # Combine: [prompt_ids, response_ids]
        full_ids = prompt_ids + response_ids

        # Add EOS token
        if self.tokenizer.eos_token_id is not None:
            full_ids = full_ids + [self.tokenizer.eos_token_id]

        # Truncate if needed
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]
            # Recalculate prompt length for mask
            prompt_len = min(len(prompt_ids), self.max_seq_length)
        else:
            prompt_len = len(prompt_ids)

        # Create input_ids and labels (shifted by 1 for next token prediction)
        # input_ids: all tokens except the last
        # labels: all tokens except the first
        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # Create response mask: 1 for response tokens in labels, 0 for prompt tokens
        # Since labels are shifted, the response starts at position (prompt_len - 1)
        response_mask = [0] * len(labels)
        response_start = max(0, prompt_len - 1)
        for i in range(response_start, len(labels)):
            response_mask[i] = 1

        # Pad to max_seq_length - 1 (since we removed one token)
        seq_len = self.max_seq_length - 1
        pad_len = seq_len - len(input_ids)

        if pad_len > 0:
            pad_token_id = self.tokenizer.pad_token_id or 0
            input_ids = input_ids + [pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 is ignored in loss
            response_mask = response_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.float),
        }


# ==============================================================================
# Loss Functions (kept for backward compatibility)
# ==============================================================================


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the SFT loss (cross-entropy) only on response tokens.

    This function computes per-token cross-entropy and averages over valid
    response tokens. For proper gradient accumulation with a constant
    normalization factor, use get_response_log_probs + sft_microbatch_train_step.

    Args:
        logits: Model logits of shape (batch_size, seq_length, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_length)
        response_mask: Mask of shape (batch_size, seq_length), 1 for response tokens

    Returns:
        Scalar loss tensor
    """
    batch_size, seq_length, vocab_size = logits.shape

    # Cast logits to float32 for stable loss computation
    logits = logits.float()

    # Flatten for cross-entropy
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute per-token cross-entropy loss (ignore_index=-100 handles padding)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fn(logits_flat, labels_flat)
    per_token_loss = per_token_loss.view(batch_size, seq_length)

    # Apply response mask (only compute loss on response tokens)
    # Also mask out any positions where labels == -100
    valid_mask = (labels != -100).float() * response_mask
    masked_loss = per_token_loss * valid_mask

    # Average over valid response tokens
    num_valid_tokens = valid_mask.sum()
    if num_valid_tokens > 0:
        loss = masked_loss.sum() / num_valid_tokens
    else:
        # Return zero loss if no valid tokens (avoid NaN)
        loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    return loss


# ==============================================================================
# Training Metrics
# ==============================================================================


@dataclass
class TrainingMetrics:
    """
    Container for tracking and visualizing training metrics.

    This class logs training metrics (loss, learning rate) at each optimizer step
    and provides utilities for saving, loading, and plotting the training curves.

    Attributes:
        steps: List of global step numbers.
        losses: List of average loss values at each step.
        learning_rates: List of learning rate values at each step.

    Example:
        >>> metrics = TrainingMetrics()
        >>> metrics.log(step=1, loss=2.5, lr=1e-5)
        >>> metrics.log(step=2, loss=2.3, lr=1e-5)
        >>> metrics.save("training_metrics.json")
        >>> metrics.plot(save_path="training_curves.png")

        # Load and plot later
        >>> loaded = TrainingMetrics.load("training_metrics.json")
        >>> loaded.plot()
    """
    steps: list[int] = None
    losses: list[float] = None
    learning_rates: list[float] = None

    def __post_init__(self):
        self.steps = []
        self.losses = []
        self.learning_rates = []

    def log(self, step: int, loss: float, lr: float):
        """Log metrics for a training step."""
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)

    def save(self, path: str):
        """Save metrics to a JSON file."""
        import json
        data = {
            "steps": self.steps,
            "losses": self.losses,
            "learning_rates": self.learning_rates,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Training metrics saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TrainingMetrics":
        """Load metrics from a JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        metrics = cls()
        metrics.steps = data["steps"]
        metrics.losses = data["losses"]
        metrics.learning_rates = data["learning_rates"]
        return metrics

    def plot(self, save_path: str | None = None, show: bool = True):
        """Plot training metrics (loss and learning rate curves)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed. Cannot plot metrics.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.steps, self.losses, 'b-', linewidth=1)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)

        # Plot learning rate
        ax2.plot(self.steps, self.learning_rates, 'r-', linewidth=1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training plot saved to {save_path}")

        if show:
            plt.show()

        plt.close()


# ==============================================================================
# Training Loop
# ==============================================================================


def train_sft(
    config: SFTConfig,
    num_samples: int | None = None,
) -> tuple[str, TrainingMetrics]:
    """
    Train a model using SFT on MATH data with automatic hardware optimization.

    This function performs supervised fine-tuning on the MATH dataset using
    the r1_zero prompt format. It automatically detects the hardware platform
    (CUDA, MPS, or CPU) and optimizes training parameters accordingly.

    Supports:
        - NVIDIA GPUs: Uses bf16/fp16 mixed precision, multi-GPU via Accelerate
        - Apple Silicon (MPS): Uses fp32 with memory-efficient settings
        - CPU: Fallback with conservative batch sizes

    Args:
        config: SFTConfig object containing training hyperparameters.
            Use -1 for batch_size, gradient_accumulation_steps, or num_workers
            to enable automatic detection based on available hardware.
            Alternatively, use SFTConfig.create_auto_config() for fully
            automatic configuration.
        num_samples: Optional limit on number of training samples. If None,
            uses all samples in the training data.

    Returns:
        tuple[str, TrainingMetrics]: A tuple containing:
            - final_model_path: Path to the saved fine-tuned model
            - metrics: TrainingMetrics object with loss/lr history

    Saves:
        - {output_dir}/final/: The fine-tuned model and tokenizer
        - {output_dir}/training_metrics.json: Training metrics (loss, lr per step)
        - {output_dir}/training_curves.png: Plot of loss and learning rate curves
        - {output_dir}/checkpoint-{step}/: Intermediate checkpoints (if save_steps > 0)

    Multi-GPU Usage:
        For multi-GPU training, launch with accelerate:
            accelerate launch scripts/run_sft.py --batch-size 4

        Or configure accelerate first:
            accelerate config  # One-time setup
            accelerate launch scripts/run_sft.py

    Example:
        >>> # Automatic hardware detection
        >>> config = SFTConfig.create_auto_config(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     train_data_path="data/math/train.jsonl",
        ... )
        >>> model_path, metrics = train_sft(config, num_samples=100)

        >>> # Manual configuration
        >>> config = SFTConfig(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     batch_size=2,
        ...     gradient_accumulation_steps=4,
        ... )
        >>> model_path, metrics = train_sft(config)
    """
    # Detect compute environment for auto-configuration (model-size aware)
    env = detect_compute_environment(config.model_name_or_path)

    # Resolve auto-detection values (-1 means auto)
    batch_size = config.batch_size if config.batch_size > 0 else env.recommended_batch_size
    grad_accum_steps = (
        config.gradient_accumulation_steps
        if config.gradient_accumulation_steps > 0
        else env.recommended_grad_accum
    )
    num_workers = config.num_workers if config.num_workers >= 0 else env.recommended_num_workers
    device = get_device(config.device)

    # Use detected environment for mixed precision and dtype
    mixed_precision = env.mixed_precision
    dtype = env.recommended_dtype

    # Log detected environment and resolved configuration
    logger.info("=" * 60)
    logger.info("COMPUTE ENVIRONMENT DETECTION")
    logger.info("=" * 60)
    logger.info(f"Platform: {env.device_name}")
    logger.info(f"Device: {env.device}")
    logger.info(f"Memory: {env.total_memory_gb:.1f} GB")
    if env.num_gpus > 1:
        logger.info(f"GPUs: {env.num_gpus}")
    logger.info(f"BF16 support: {env.supports_bf16}")
    logger.info(f"FP16 support: {env.supports_fp16}")
    logger.info("")
    logger.info("RESOLVED TRAINING CONFIGURATION")
    logger.info("-" * 40)
    logger.info(f"Batch size: {batch_size}" + (" (auto-detected)" if config.batch_size < 0 else ""))
    logger.info(f"Gradient accumulation: {grad_accum_steps}" + (" (auto-detected)" if config.gradient_accumulation_steps < 0 else ""))
    logger.info(f"Num workers: {num_workers}" + (" (auto-detected)" if config.num_workers < 0 else ""))
    logger.info(f"Mixed precision: {mixed_precision}")
    logger.info(f"Model dtype: {dtype}")
    logger.info("=" * 60)

    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision=mixed_precision,
    )

    # Log accelerator info
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Num processes: {accelerator.num_processes}")

    # Load tokenizer (only log on main process)
    if accelerator.is_main_process:
        logger.info(f"Loading tokenizer from {config.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if accelerator.is_main_process:
        logger.info(f"Loading model from {config.model_name_or_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.train()

    # Create dataset and dataloader
    dataset = MathSFTDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        num_samples=num_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use resolved batch_size
        shuffle=True,
        num_workers=num_workers,  # Use resolved num_workers
        pin_memory=(device == "cuda"),
    )

    # Calculate training steps
    # Note: with accelerate, steps per epoch = len(dataloader) / num_processes
    total_steps_per_device = len(dataloader) * config.num_epochs
    effective_batch_size = batch_size * grad_accum_steps * accelerator.num_processes
    num_update_steps = total_steps_per_device // grad_accum_steps
    warmup_steps = int(num_update_steps * config.warmup_ratio)

    if accelerator.is_main_process:
        logger.info(f"Training configuration:")
        logger.info(f"  Total examples: {len(dataset)}")
        logger.info(f"  Batch size per device: {batch_size}")
        logger.info(f"  Num devices: {accelerator.num_processes}")
        logger.info(f"  Gradient accumulation steps: {grad_accum_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Total training steps per device: {total_steps_per_device}")
        logger.info(f"  Update steps: {num_update_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Create output directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    # Wait for directory creation
    accelerator.wait_for_everyone()

    # Initialize metrics tracking
    metrics = TrainingMetrics()

    # Training loop
    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(config.num_epochs):
        if accelerator.is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")

        # Only show progress bar on main process
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_main_process,
        )

        for step, batch in enumerate(progress_bar):
            # Batch is already on correct device via accelerator.prepare
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            response_mask = batch["response_mask"]

            # Use accelerator's gradient accumulation context
            with accelerator.accumulate(model):
                # Forward pass to get log probabilities
                log_prob_result = get_response_log_probs(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False,
                )
                policy_log_probs = log_prob_result["log_probs"]

                # Compute SFT loss (negative log likelihood)
                num_response_tokens = response_mask.sum()
                normalize_constant = num_response_tokens if num_response_tokens > 0 else 1.0

                # NLL loss normalized by response tokens
                nll_loss = masked_normalize(
                    tensor=-policy_log_probs,
                    mask=response_mask,
                    normalize_constant=normalize_constant,
                    dim=None,
                )

                # Backward pass (accelerator handles gradient accumulation scaling)
                accelerator.backward(nll_loss)

                # Clip gradients (only when actually updating)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Track loss
            accumulated_loss += nll_loss.detach().item()

            # Update progress bar
            if accelerator.is_main_process:
                progress_bar.set_postfix({
                    "step_loss": f"{nll_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

            # Check if we completed a gradient accumulation cycle
            if accelerator.sync_gradients:
                global_step += 1

                # Compute average loss over the gradient accumulation window
                avg_loss = accumulated_loss / grad_accum_steps
                current_lr = scheduler.get_last_lr()[0]
                accumulated_loss = 0.0

                # Log metrics (only on main process)
                if accelerator.is_main_process:
                    metrics.log(step=global_step, loss=avg_loss, lr=current_lr)

                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    })

                    if global_step % config.logging_steps == 0:
                        logger.info(
                            f"Step {global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}"
                        )

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, tokenizer, checkpoint_dir)
                        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    accelerator.wait_for_everyone()
    final_model_path = os.path.join(config.output_dir, "final")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(unwrapped_model, tokenizer, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")

        # Save training metrics
        metrics_path = os.path.join(config.output_dir, "training_metrics.json")
        metrics.save(metrics_path)

        # Save training plot
        plot_path = os.path.join(config.output_dir, "training_curves.png")
        try:
            metrics.plot(save_path=plot_path, show=False)
        except Exception as e:
            logger.warning(f"Could not save training plot: {e}")

    return final_model_path, metrics


def save_model(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
) -> None:
    """Save model and tokenizer to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model and tokenizer saved to {output_dir}")
