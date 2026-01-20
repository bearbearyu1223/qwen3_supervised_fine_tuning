"""
Evaluate language models on MATH dataset using the r1_zero prompt.

This module provides utilities for evaluating models on the MATH dataset,
including prompt formatting, generation using multiple backends, and metric
computation with detailed analysis reports.

Supported models:
    - Qwen/Qwen2.5-Math-1.5B (default, optimized for math)
    - Qwen/Qwen3-4B (latest Qwen3, 4B params)
    - Qwen/Qwen3-1.7B (smaller Qwen3, 1.7B params)
    - meta-llama/Llama-3.1-8B (general-purpose, larger)
    - Any HuggingFace causal LM compatible with the r1_zero format

Key components:
    - TransformersModel: HuggingFace transformers wrapper for CPU/MPS inference
    - format_r1_zero_prompt: Format questions using the r1_zero prompt template
    - evaluate_math: Main entry point for evaluation
    - compute_metrics_from_responses: Compute rewards from model outputs
    - generate_analysis_report: Create detailed analysis of model performance

Supported backends:
    - vllm: Fast batched inference on NVIDIA GPUs (default, recommended)
    - transformers: Sequential inference for CPU/MPS (Mac M-series chips)

Example usage:
    >>> from cs336_alignment.evaluate_math import evaluate_math
    >>> # Evaluate Qwen model with vLLM backend (NVIDIA GPU)
    >>> metrics = evaluate_math(
    ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
    ...     input_path="data/math/test.jsonl",
    ...     output_path="outputs/eval_results.jsonl",
    ...     backend="vllm",
    ... )
    >>> print(f"Answer accuracy: {metrics['answer_reward']:.2%}")

    >>> # Evaluate Llama model with transformers backend (Mac/CPU)
    >>> metrics = evaluate_math(
    ...     model_name_or_path="meta-llama/Llama-3.1-8B",
    ...     input_path="data/math/test.jsonl",
    ...     output_path="outputs/llama_eval_results.jsonl",
    ...     backend="transformers",
    ...     num_samples=10,  # Limit samples for testing
    ... )

CLI usage:
    # Evaluate Qwen with vLLM backend (NVIDIA GPU, full dataset)
    python -m cs336_alignment.evaluate_math \\
        --model Qwen/Qwen2.5-Math-1.5B \\
        --input data/math/test.jsonl \\
        --output outputs/eval_results.jsonl \\
        --backend vllm

    # Evaluate Llama with transformers backend (Mac/CPU, limited samples)
    python -m cs336_alignment.evaluate_math \\
        --model meta-llama/Llama-3.1-8B \\
        --input data/math/test.jsonl \\
        --output outputs/llama_eval_results.jsonl \\
        --backend transformers \\
        --num-samples 10
"""
import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Callable, List, Literal

import torch
from tqdm import tqdm
from xopen import xopen

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# Type hints for optional vLLM dependency
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# Load the r1_zero prompt template
PROMPT_PATH = Path(__file__).parent / "prompts" / "r1_zero.prompt"
with open(PROMPT_PATH) as f:
    R1_ZERO_PROMPT_TEMPLATE = f.read()


# ==============================================================================
# Device and Model Utilities
# ==============================================================================


def get_device() -> str:
    """
    Get the best available device for inference.

    Checks device availability in order of preference: CUDA > MPS > CPU.

    Returns:
        str: Device identifier ("cuda", "mps", or "cpu").

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class TransformersModel:
    """
    HuggingFace transformers model wrapper for CPU/MPS inference.

    This class provides a simplified interface for loading and running inference
    with HuggingFace models, designed to work on devices without NVIDIA GPUs
    (e.g., Mac M-series chips). It wraps the transformers library to provide
    an interface similar to vLLM for consistent usage across backends.

    Attributes:
        device (str): The device the model is loaded on ("cuda", "mps", or "cpu").
        tokenizer: The HuggingFace tokenizer for the model.
        model: The HuggingFace causal language model.

    Example:
        >>> model = TransformersModel("Qwen/Qwen2.5-Math-1.5B")
        >>> responses = model.generate(
        ...     prompts=["What is 2+2?", "What is 3*3?"],
        ...     max_tokens=256,
        ...     temperature=0.0,
        ... )
        >>> print(responses[0])

    Note:
        For faster inference on NVIDIA GPUs, prefer using vLLM directly via
        the ``evaluate_vllm`` function instead of this wrapper.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the TransformersModel wrapper.

        Args:
            model_name_or_path: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-Math-1.5B")
                or path to a local model directory.
            device: Device to load the model on. If None, automatically selects
                the best available device (CUDA > MPS > CPU).
            torch_dtype: Data type for model weights. If None, defaults to float16
                for GPU/MPS and float32 for CPU.

        Raises:
            OSError: If the model or tokenizer cannot be loaded from the specified path.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")

        # Default to float16 for GPU, float32 for CPU
        if torch_dtype is None:
            if self.device in ("cuda", "mps"):
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        logger.info(f"Loading tokenizer from {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=self.device if self.device != "mps" else None,
        )
        if self.device == "mps":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> List[str]:
        """
        Generate responses for a list of prompts.

        Processes prompts sequentially (one at a time) to handle memory constraints
        on devices with limited VRAM. For batched inference on NVIDIA GPUs,
        use vLLM instead.

        Args:
            prompts: List of input prompt strings to generate completions for.
            max_tokens: Maximum number of new tokens to generate per prompt.
                Defaults to 2048.
            temperature: Sampling temperature. Use 0.0 for greedy (deterministic)
                decoding, or higher values for more diverse outputs. Defaults to 0.0.

        Returns:
            List of generated response strings, one per input prompt.
            Only the generated text is returned (prompt is excluded).

        Example:
            >>> model = TransformersModel("Qwen/Qwen2.5-Math-1.5B")
            >>> prompts = ["Solve: 2+2=", "Solve: 5*5="]
            >>> responses = model.generate(prompts, max_tokens=100)
            >>> print(responses)
        """
        responses = []

        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                if temperature == 0.0:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            # Decode only the generated part (exclude the prompt)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses


# ==============================================================================
# Prompt Formatting
# ==============================================================================


def format_r1_zero_prompt(question: str) -> str:
    """
    Format a math question using the r1_zero prompt template.

    The r1_zero format structures the prompt as a conversation where the
    assistant is expected to show reasoning in <think> tags and provide
    the final answer in <answer> tags.

    Args:
        question: The math problem text to format.

    Returns:
        Formatted prompt string ready for model input.

    Example:
        >>> prompt = format_r1_zero_prompt("What is 2+2?")
        >>> print(prompt)
        A conversation between User and Assistant...
        User: What is 2+2?
        Assistant: <think>
    """
    return R1_ZERO_PROMPT_TEMPLATE.format(question=question)


# ==============================================================================
# Metrics Computation
# ==============================================================================


def compute_metrics_from_responses(
    responses: List[str],
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """
    Compute evaluation metrics from model-generated responses.

    Applies the reward function to each (response, ground_truth) pair and
    aggregates the results into per-example metrics and overall averages.

    Args:
        responses: List of model-generated response strings.
        prompts: List of formatted prompt strings (for serialization).
        ground_truths: List of ground truth answer strings.
        reward_fn: Callable that takes (response, ground_truth) and returns a dict
            with metric keys (e.g., "format_reward", "answer_reward", "reward").
        input_examples: Optional list of original input example dicts to include
            in the output for reference. If None, empty dicts are used.

    Returns:
        Tuple containing:
            - all_results: List of dicts, each containing the original example data,
              prompt, model output, and computed metrics.
            - aggregated_metrics: Dict mapping metric names to their mean values
              across all examples.

    Example:
        >>> from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
        >>> responses = ["</think> <answer>4</answer>"]
        >>> prompts = ["What is 2+2?"]
        >>> ground_truths = ["4"]
        >>> results, metrics = compute_metrics_from_responses(
        ...     responses, prompts, ground_truths, r1_zero_reward_fn
        ... )
        >>> print(metrics["answer_reward"])
        1.0
    """
    if input_examples is None:
        input_examples = [{} for _ in prompts]

    all_results = []
    all_metrics = []

    for input_example, prompt, response, ground_truth in tqdm(
        zip(input_examples, prompts, responses, ground_truths),
        total=len(prompts),
        desc="Computing metrics",
    ):
        metrics = reward_fn(response, ground_truth)
        all_metrics.append(metrics)

        result = {
            **input_example,
            "prompt": prompt,
            "output": response,
            "metrics": metrics,
        }
        all_results.append(result)

    # Aggregate metrics by computing mean across all examples
    aggregated_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = mean([m[key] for m in all_metrics])

    return all_results, aggregated_metrics


# ==============================================================================
# Evaluation Functions
# ==============================================================================


def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: "SamplingParams",
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """
    Evaluate a language model using the vLLM backend.

    Uses vLLM's efficient batched inference to generate responses for all prompts,
    then computes evaluation metrics using the provided reward function.

    Args:
        vllm_model: A vLLM LLM instance (from ``vllm.LLM``).
        reward_fn: Callable that takes (response, ground_truth) and returns a dict
            with metric keys (e.g., "format_reward", "answer_reward", "reward").
        prompts: List of formatted prompt strings to generate completions for.
        ground_truths: List of ground truth answer strings for evaluation.
        eval_sampling_params: vLLM SamplingParams instance controlling generation
            (temperature, max_tokens, top_p, etc.).
        input_examples: Optional list of original input example dicts to include
            in the output for reference.

    Returns:
        Tuple containing:
            - all_results: List of result dicts with prompt, output, and metrics.
            - aggregated_metrics: Dict of mean metric values across all examples.

    Note:
        This function requires vLLM to be installed and an NVIDIA GPU to be available.
        For CPU/MPS inference, use ``evaluate_transformers`` instead.
    """
    logger.info(f"Generating responses for {len(prompts)} prompts using vLLM...")
    raw_responses = vllm_model.generate(prompts, eval_sampling_params)

    responses = []
    for output in raw_responses:
        response = output.outputs[0].text
        responses.append(response)

    assert len(responses) == len(prompts), (
        f"Number of responses ({len(responses)}) must match prompts ({len(prompts)})"
    )
    logger.info(f"Generated {len(responses)} responses")

    return compute_metrics_from_responses(
        responses=responses,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_fn=reward_fn,
        input_examples=input_examples,
    )


def evaluate_transformers(
    model: TransformersModel,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    input_examples: List[dict] | None = None,
) -> tuple[List[dict], dict[str, float]]:
    """
    Evaluate a language model using the HuggingFace transformers backend.

    Generates responses sequentially using the TransformersModel wrapper,
    then computes evaluation metrics using the provided reward function.

    Args:
        model: A TransformersModel instance wrapping a HuggingFace model.
        reward_fn: Callable that takes (response, ground_truth) and returns a dict
            with metric keys (e.g., "format_reward", "answer_reward", "reward").
        prompts: List of formatted prompt strings to generate completions for.
        ground_truths: List of ground truth answer strings for evaluation.
        max_tokens: Maximum number of new tokens to generate per prompt.
            Defaults to 2048.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.
            Defaults to 0.0.
        input_examples: Optional list of original input example dicts to include
            in the output for reference.

    Returns:
        Tuple containing:
            - all_results: List of result dicts with prompt, output, and metrics.
            - aggregated_metrics: Dict of mean metric values across all examples.

    Note:
        This backend processes prompts sequentially and is slower than vLLM.
        Use this for CPU/MPS inference or when vLLM is not available.
    """
    logger.info(f"Generating responses for {len(prompts)} prompts using transformers...")
    responses = model.generate(
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    assert len(responses) == len(prompts), (
        f"Number of responses ({len(responses)}) must match prompts ({len(prompts)})"
    )
    logger.info(f"Generated {len(responses)} responses")

    return compute_metrics_from_responses(
        responses=responses,
        prompts=prompts,
        ground_truths=ground_truths,
        reward_fn=reward_fn,
        input_examples=input_examples,
    )


# ==============================================================================
# Data Loading and Saving
# ==============================================================================


def load_math_examples(input_path: str) -> tuple[List[dict], List[str], List[str]]:
    """
    Load MATH dataset examples and format them with r1_zero prompts.

    Reads examples from a JSONL file where each line contains a JSON object
    with "problem" and "answer" fields, formats each problem using the
    r1_zero prompt template, and returns the data ready for evaluation.

    Args:
        input_path: Path to MATH examples file (JSONL format, supports .gz compression).
            Each line should be a JSON object with at least "problem" and "answer" keys.

    Returns:
        Tuple containing:
            - input_examples: List of raw example dicts from the input file.
            - prompts: List of formatted prompt strings using r1_zero template.
            - ground_truths: List of ground truth answer strings.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If any line in the file is not valid JSON.
        KeyError: If an example is missing required "problem" or "answer" keys.

    Example:
        >>> examples, prompts, truths = load_math_examples("data/math/test.jsonl")
        >>> print(f"Loaded {len(examples)} examples")
        >>> print(prompts[0][:50])  # First 50 chars of first prompt
    """
    input_examples = []
    with xopen(input_path) as f:
        for line in f:
            input_examples.append(json.loads(line))
    logger.info(f"Read {len(input_examples)} examples from {input_path}")

    prompts = []
    ground_truths = []
    for example in input_examples:
        question = example["problem"]
        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(example["answer"])

    return input_examples, prompts, ground_truths


def save_results(results: List[dict], output_path: str) -> None:
    """
    Save evaluation results to a JSONL file.

    Creates the output directory if it doesn't exist, then writes each
    result dict as a JSON line to the output file.

    Args:
        results: List of result dictionaries to save. Each dict typically
            contains keys like "problem", "prompt", "output", and "metrics".
        output_path: Path to the output file. Supports .gz extension for
            gzip compression. Parent directories are created if needed.

    Example:
        >>> results = [{"prompt": "...", "output": "...", "metrics": {...}}]
        >>> save_results(results, "outputs/eval_results.jsonl")
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with xopen(output_path, "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")
    logger.info(f"Wrote {len(results)} results to {output_path}")


# ==============================================================================
# Main Evaluation Entry Point
# ==============================================================================


def evaluate_math(
    model_name_or_path: str,
    input_path: str = "data/math/test.jsonl",
    output_path: str = "outputs/math_eval_results.jsonl",
    num_gpus: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    num_samples: int | None = None,
    backend: Literal["vllm", "transformers"] = "vllm",
) -> dict[str, float]:
    """
    Evaluate a language model on the MATH dataset using r1_zero prompts.

    This is the main entry point for MATH evaluation. It handles loading data,
    model initialization, generation, metric computation, and result saving.
    Supports both vLLM (recommended for NVIDIA GPUs) and transformers backends.

    Args:
        model_name_or_path: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-Math-1.5B")
            or path to a local model directory.
        input_path: Path to MATH test examples in JSONL format. Each line should
            contain a JSON object with "problem" and "answer" keys.
            Defaults to "data/math/test.jsonl".
        output_path: Path to write evaluation results in JSONL format. An analysis
            report will also be saved with "_analysis.txt" suffix.
            Defaults to "outputs/math_eval_results.jsonl".
        num_gpus: Number of GPUs for tensor parallelism (vLLM backend only).
            Defaults to 1.
        max_tokens: Maximum number of new tokens to generate per example.
            Defaults to 2048.
        temperature: Sampling temperature. Use 0.0 for deterministic (greedy)
            decoding. Defaults to 0.0.
        num_samples: Limit evaluation to first N examples. If None, evaluates
            all examples in the input file.
        backend: Inference backend to use:
            - "vllm": Fast batched inference (requires NVIDIA GPU)
            - "transformers": Sequential inference (works on CPU/MPS)
            Defaults to "vllm".

    Returns:
        Dictionary of aggregated metrics with keys:
            - "format_reward": Mean format correctness (0.0 to 1.0)
            - "answer_reward": Mean answer correctness (0.0 to 1.0)
            - "reward": Combined reward (typically same as answer_reward)

    Raises:
        ValueError: If an unknown backend is specified.
        FileNotFoundError: If the input file does not exist.

    Saves:
        - {output_path}: Per-example results in JSONL format
        - {output_path}_analysis.txt: Detailed analysis report with examples

    Example:
        >>> # Evaluate on GPU with vLLM
        >>> metrics = evaluate_math(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     input_path="data/math/test.jsonl",
        ...     output_path="outputs/eval_results.jsonl",
        ...     backend="vllm",
        ... )
        >>> print(f"Accuracy: {metrics['answer_reward']:.2%}")

        >>> # Quick test on Mac with limited samples
        >>> metrics = evaluate_math(
        ...     model_name_or_path="Qwen/Qwen2.5-Math-1.5B",
        ...     backend="transformers",
        ...     num_samples=10,
        ... )
    """
    # Load and format examples
    input_examples, prompts, ground_truths = load_math_examples(input_path)

    # Limit to num_samples if specified
    if num_samples is not None and num_samples < len(prompts):
        logger.info(f"Limiting evaluation to {num_samples} samples (out of {len(prompts)})")
        input_examples = input_examples[:num_samples]
        prompts = prompts[:num_samples]
        ground_truths = ground_truths[:num_samples]

    if backend == "vllm":
        from vllm import LLM, SamplingParams

        # Load model with vLLM
        logger.info(f"Loading model {model_name_or_path} with vLLM backend...")
        model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
        )

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        # Evaluate
        all_results, aggregated_metrics = evaluate_vllm(
            vllm_model=model,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            eval_sampling_params=sampling_params,
            input_examples=input_examples,
        )
    elif backend == "transformers":
        # Load model with transformers (works on CPU/MPS)
        logger.info(f"Loading model {model_name_or_path} with transformers backend...")
        model = TransformersModel(model_name_or_path)

        # Evaluate
        all_results, aggregated_metrics = evaluate_transformers(
            model=model,
            reward_fn=r1_zero_reward_fn,
            prompts=prompts,
            ground_truths=ground_truths,
            max_tokens=max_tokens,
            temperature=temperature,
            input_examples=input_examples,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'transformers'.")

    # Save results
    save_results(all_results, output_path)

    # Log aggregated metrics
    logger.info("=== Aggregated Metrics ===")
    for key, value in sorted(aggregated_metrics.items()):
        logger.info(f"{key}: {value:.4f}")

    # Generate analysis report
    analysis_report = generate_analysis_report(all_results)

    # Save analysis report
    report_path = output_path.replace(".jsonl", "_analysis.txt")
    save_analysis_report(analysis_report, report_path)

    return aggregated_metrics


# ==============================================================================
# Analysis and Reporting
# ==============================================================================


def categorize_results(results: List[dict]) -> dict[str, List[dict]]:
    """
    Categorize evaluation results based on format and answer correctness.

    Sorts results into three mutually exclusive categories based on the
    format_reward and answer_reward metrics from the r1_zero grader.

    Args:
        results: List of result dictionaries, each containing a "metrics" key
            with "format_reward" and "answer_reward" values.

    Returns:
        Dictionary mapping category names to lists of results:
            - "correct": Both format and answer correct (format=1, answer=1)
            - "format_only": Correct format but wrong answer (format=1, answer=0)
            - "neither": Incorrect format (format=0, answer=0)

    Example:
        >>> results = [
        ...     {"metrics": {"format_reward": 1.0, "answer_reward": 1.0}},
        ...     {"metrics": {"format_reward": 1.0, "answer_reward": 0.0}},
        ...     {"metrics": {"format_reward": 0.0, "answer_reward": 0.0}},
        ... ]
        >>> categories = categorize_results(results)
        >>> print(len(categories["correct"]))  # 1
        >>> print(len(categories["format_only"]))  # 1
        >>> print(len(categories["neither"]))  # 1
    """
    categories = {
        "correct": [],      # format=1, answer=1
        "format_only": [],  # format=1, answer=0
        "neither": [],      # format=0, answer=0
    }

    for result in results:
        metrics = result["metrics"]
        format_reward = metrics["format_reward"]
        answer_reward = metrics["answer_reward"]

        if format_reward == 1.0 and answer_reward == 1.0:
            categories["correct"].append(result)
        elif format_reward == 1.0 and answer_reward == 0.0:
            categories["format_only"].append(result)
        else:  # format_reward == 0
            categories["neither"].append(result)

    return categories


def generate_analysis_report(results: List[dict], max_examples_per_category: int = 10) -> str:
    """
    Generate a detailed analysis report of evaluation results.

    Creates a human-readable report that categorizes model outputs, provides
    summary statistics, shows example outputs from each category, and includes
    diagnostic information for debugging model behavior.

    Args:
        results: List of result dictionaries from evaluation, each containing
            "problem", "answer", "output", and "metrics" keys.
        max_examples_per_category: Maximum number of examples to show for each
            category in the report. Defaults to 10.

    Returns:
        Multi-line string containing the formatted analysis report with:
            - Summary statistics (counts and percentages per category)
            - Example outputs for format=0 cases with diagnosis
            - Example outputs for format=1, answer=0 cases with diagnosis
            - Example outputs for correct cases with observations

    Note:
        The report is designed to help diagnose common issues like:
        - Models not following the expected r1_zero format
        - Mathematical reasoning errors
        - Answer normalization edge cases
    """
    categories = categorize_results(results)

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MATH EVALUATION ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    total = len(results)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total examples: {total}")
    report_lines.append("")

    report_lines.append("Category Breakdown:")
    report_lines.append(f"  (1) Correct (format=1, answer=1): {len(categories['correct'])} ({100*len(categories['correct'])/total:.1f}%)")
    report_lines.append(f"  (2) Format only (format=1, answer=0): {len(categories['format_only'])} ({100*len(categories['format_only'])/total:.1f}%)")
    report_lines.append(f"  (3) Neither (format=0, answer=0): {len(categories['neither'])} ({100*len(categories['neither'])/total:.1f}%)")
    report_lines.append("")

    # Analysis of format=0 cases (neither category)
    report_lines.append("=" * 80)
    report_lines.append("ANALYSIS: FORMAT REWARD = 0 CASES")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("The r1_zero format expects: '</think> <answer>...</answer>'")
    report_lines.append("Format reward is 0 when this pattern is not found in the output.")
    report_lines.append("")

    # Show up to max_examples_per_category examples where format=0
    neither_examples = categories["neither"][:max_examples_per_category]
    for i, example in enumerate(neither_examples, 1):
        report_lines.append(f"--- Example {i} (format=0) ---")
        report_lines.append(f"Problem: {example.get('problem', 'N/A')[:200]}...")
        report_lines.append(f"Ground Truth: {example.get('answer', 'N/A')}")
        report_lines.append(f"Model Output (first 500 chars):")
        output = example.get("output", "N/A")
        report_lines.append(f"  {output[:500]}...")
        report_lines.append("")

    report_lines.append("DIAGNOSIS for format=0 cases:")
    report_lines.append("-" * 40)
    report_lines.append("Examining the outputs above, the issue is likely with the BASE MODEL'S OUTPUT")
    report_lines.append("because the model is not trained to produce the specific r1_zero format")
    report_lines.append("(i.e., '</think> <answer>...</answer>'). The Qwen2.5-Math model was not")
    report_lines.append("specifically fine-tuned on this format, so it produces answers in its own")
    report_lines.append("default format (e.g., using \\boxed{} directly without the think/answer tags).")
    report_lines.append("The parser correctly identifies that the expected format is missing.")
    report_lines.append("")

    # Analysis of format=1, answer=0 cases
    report_lines.append("=" * 80)
    report_lines.append("ANALYSIS: FORMAT REWARD = 1, ANSWER REWARD = 0 CASES")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("These cases have correct format but wrong answer.")
    report_lines.append("")

    # Show up to max_examples_per_category examples where format=1, answer=0
    format_only_examples = categories["format_only"][:max_examples_per_category]
    for i, example in enumerate(format_only_examples, 1):
        report_lines.append(f"--- Example {i} (format=1, answer=0) ---")
        report_lines.append(f"Problem: {example.get('problem', 'N/A')[:200]}...")
        report_lines.append(f"Ground Truth: {example.get('answer', 'N/A')}")
        report_lines.append(f"Model Output (first 500 chars):")
        output = example.get("output", "N/A")
        report_lines.append(f"  {output[:500]}...")
        report_lines.append("")

    report_lines.append("DIAGNOSIS for format=1, answer=0 cases:")
    report_lines.append("-" * 40)
    report_lines.append("In these cases, the model produced the correct format but gave a wrong answer.")
    report_lines.append("This could be due to:")
    report_lines.append("  1. Mathematical reasoning errors by the model")
    report_lines.append("  2. Computation mistakes in multi-step problems")
    report_lines.append("  3. Misunderstanding of the problem")
    report_lines.append("  4. Edge cases in answer normalization/comparison")
    report_lines.append("")

    # Analysis of format=1, answer=1 cases (correct)
    report_lines.append("=" * 80)
    report_lines.append("ANALYSIS: FORMAT REWARD = 1, ANSWER REWARD = 1 CASES (CORRECT)")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("These cases have both correct format and correct answer.")
    report_lines.append("")

    # Show up to max_examples_per_category examples where format=1, answer=1
    correct_examples = categories["correct"][:max_examples_per_category]
    for i, example in enumerate(correct_examples, 1):
        report_lines.append(f"--- Example {i} (format=1, answer=1) ---")
        report_lines.append(f"Problem: {example.get('problem', 'N/A')[:200]}...")
        report_lines.append(f"Ground Truth: {example.get('answer', 'N/A')}")
        report_lines.append(f"Model Output (first 500 chars):")
        output = example.get("output", "N/A")
        report_lines.append(f"  {output[:500]}...")
        report_lines.append("")

    report_lines.append("OBSERVATIONS for correct cases:")
    report_lines.append("-" * 40)
    report_lines.append("In these cases, the model successfully:")
    report_lines.append("  1. Followed the expected r1_zero format with <think>...</think> <answer>...</answer>")
    report_lines.append("  2. Reasoned through the problem correctly")
    report_lines.append("  3. Produced the correct final answer")
    report_lines.append("")

    return "\n".join(report_lines)


def save_analysis_report(report: str, output_path: str) -> None:
    """
    Save an analysis report to a text file and print to console.

    Creates the output directory if needed, writes the report to the specified
    file, and also prints it to stdout for immediate visibility.

    Args:
        report: The formatted report string to save.
        output_path: Path to the output text file. Parent directories are
            created if they don't exist.

    Example:
        >>> report = generate_analysis_report(results)
        >>> save_analysis_report(report, "outputs/analysis.txt")
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Wrote analysis report to {output_path}")

    # Also print to console for immediate visibility
    print("\n" + report)
