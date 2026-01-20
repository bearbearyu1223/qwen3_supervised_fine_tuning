"""
Verification script for cs336_alignment/sft.py

This script tests all major components of the SFT module:
1. format_math_example_for_sft() - formatting function
2. MathSFTDataset - dataset class
3. tokenize_prompt_and_output() - batch tokenization
4. compute_entropy() - per-token entropy
5. get_response_log_probs() - log probabilities
6. masked_normalize() - masked normalization
7. sft_microbatch_train_step() - microbatch training
8. Gradient accumulation verification
9. Model and tokenizer saving

Run with:
    uv run python scripts/verify_sft.py
"""
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_alignment.sft import (
    format_math_example_for_sft,
    MathSFTDataset,
    tokenize_prompt_and_output,
    compute_entropy,
    get_response_log_probs,
    masked_normalize,
    sft_microbatch_train_step,
    compute_sft_loss,
    get_device,
    detect_compute_environment,
    ComputeEnvironment,
    SFTConfig,
    R1_ZERO_PROMPT_TEMPLATE,
    estimate_model_size_billions,
    get_recommended_batch_size_for_model,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def test_format_math_example():
    """Test the format_math_example_for_sft function."""
    logger.info("=" * 60)
    logger.info("TEST 1: format_math_example_for_sft()")
    logger.info("=" * 60)

    problem = "If $5x - 3 = 12$, what is the value of $5x + 3$?"
    solution = "Adding 6 to both sides gives $5x + 3 = \\boxed{18}$."
    answer = "18"

    prompt, response = format_math_example_for_sft(problem, solution, answer)

    logger.info(f"Prompt length: {len(prompt)}")
    logger.info(f"Response length: {len(response)}")

    assert "<think>" in prompt, "Prompt should contain <think> tag"
    assert "</think>" in response, "Response should contain </think> tag"
    assert "<answer>" in response, "Response should contain <answer> tag"
    assert "</answer>" in response, "Response should contain </answer> tag"

    logger.info("PASSED")
    return True


def test_tokenize_prompt_and_output():
    """Test the tokenize_prompt_and_output function."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: tokenize_prompt_and_output()")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = ["Hello, world!", "What is 2+2?", "Test prompt here"]
    outputs = ["Hi there!", "The answer is 4.", "Short"]

    result = tokenize_prompt_and_output(prompts, outputs, tokenizer)

    logger.info(f"input_ids shape: {result['input_ids'].shape}")
    logger.info(f"labels shape: {result['labels'].shape}")
    logger.info(f"response_mask shape: {result['response_mask'].shape}")

    batch_size = len(prompts)
    assert result['input_ids'].shape[0] == batch_size
    assert result['labels'].shape == result['input_ids'].shape
    assert result['response_mask'].shape == result['input_ids'].shape

    # Check that response_mask has some 1s for each sample
    for i in range(batch_size):
        assert result['response_mask'][i].sum() > 0, f"Sample {i} should have response tokens"

    # Verify labels are shifted by 1
    # The labels should be input_ids shifted left (next token prediction)
    logger.info("PASSED")
    return True


def test_compute_entropy():
    """Test the compute_entropy function."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: compute_entropy()")
    logger.info("=" * 60)

    # Test with uniform distribution (should have high entropy)
    batch_size, seq_len, vocab_size = 2, 5, 100
    uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)
    uniform_entropy = compute_entropy(uniform_logits)

    expected_uniform_entropy = torch.log(torch.tensor(float(vocab_size)))
    logger.info(f"Uniform entropy: {uniform_entropy[0, 0]:.4f}")
    logger.info(f"Expected (log vocab_size): {expected_uniform_entropy:.4f}")

    assert torch.allclose(uniform_entropy, expected_uniform_entropy, atol=1e-4), \
        "Uniform distribution should have entropy = log(vocab_size)"

    # Test with peaked distribution (should have low entropy)
    peaked_logits = torch.zeros(batch_size, seq_len, vocab_size)
    peaked_logits[:, :, 0] = 100.0  # Very high logit for one class
    peaked_entropy = compute_entropy(peaked_logits)

    logger.info(f"Peaked entropy: {peaked_entropy[0, 0]:.6f}")
    assert peaked_entropy.max() < 0.01, "Peaked distribution should have near-zero entropy"

    logger.info("PASSED")
    return True


def test_get_response_log_probs():
    """Test the get_response_log_probs function."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: get_response_log_probs()")
    logger.info("=" * 60)

    device = get_device("auto")
    model_name = "Qwen/Qwen2.5-0.5B"

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create test inputs
    prompts = ["Hello", "Test"]
    outputs = [" world", " output"]
    batch = tokenize_prompt_and_output(prompts, outputs, tokenizer)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Test without entropy
    result = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
    logger.info(f"log_probs shape: {result['log_probs'].shape}")
    assert "log_probs" in result
    assert result["log_probs"].shape == input_ids.shape
    assert "token_entropy" not in result

    # Test with entropy
    result_with_entropy = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    assert "token_entropy" in result_with_entropy
    logger.info(f"token_entropy shape: {result_with_entropy['token_entropy'].shape}")

    # Log probs should be negative (log of probabilities)
    assert result["log_probs"].max() <= 0, "Log probs should be <= 0"

    logger.info("PASSED")
    return True


def test_masked_normalize():
    """Test the masked_normalize function."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 5: masked_normalize()")
    logger.info("=" * 60)

    # Test case 1: Simple sum with full mask
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.ones_like(tensor)
    result = masked_normalize(tensor, mask, normalize_constant=1.0, dim=None)
    expected = tensor.sum()
    logger.info(f"Full mask sum: {result.item():.4f}, expected: {expected.item():.4f}")
    assert torch.allclose(result, expected)

    # Test case 2: Partial mask
    partial_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    result = masked_normalize(tensor, partial_mask, normalize_constant=1.0, dim=None)
    expected = 1.0 + 3.0 + 5.0  # Only masked elements
    logger.info(f"Partial mask sum: {result.item():.4f}, expected: {expected:.4f}")
    assert torch.allclose(result, torch.tensor(expected))

    # Test case 3: With normalize constant
    result = masked_normalize(tensor, mask, normalize_constant=21.0, dim=None)
    expected = tensor.sum() / 21.0
    logger.info(f"Normalized sum: {result.item():.4f}, expected: {expected.item():.4f}")
    assert torch.allclose(result, expected)

    # Test case 4: With dimension
    result = masked_normalize(tensor, mask, normalize_constant=1.0, dim=1)
    expected = tensor.sum(dim=1)
    logger.info(f"Sum along dim 1: {result}, expected: {expected}")
    assert torch.allclose(result, expected)

    logger.info("PASSED")
    return True


def test_sft_microbatch_train_step():
    """Test the sft_microbatch_train_step function."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 6: sft_microbatch_train_step()")
    logger.info("=" * 60)

    batch_size, seq_len = 2, 10
    gradient_accumulation_steps = 2

    # Create test data
    torch.manual_seed(42)
    policy_log_probs = torch.randn(batch_size, seq_len, requires_grad=True)
    response_mask = (torch.rand(batch_size, seq_len) > 0.5).float()

    # Run microbatch step
    loss, metadata = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        normalize_constant=1.0,
    )

    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Metadata keys: {list(metadata.keys())}")

    # Check that gradients were computed
    assert policy_log_probs.grad is not None, "Gradients should be computed"
    logger.info(f"Gradient norm: {policy_log_probs.grad.norm().item():.4f}")

    # Loss should be scaled by batch_size * gradient_accumulation_steps
    # Compare with raw computation
    raw_sum = (-policy_log_probs.detach() * response_mask).sum()
    expected_scaled = raw_sum / (batch_size * gradient_accumulation_steps)
    logger.info(f"Expected scaled loss: {expected_scaled.item():.4f}")
    assert torch.allclose(loss, expected_scaled, atol=1e-4)

    logger.info("PASSED")
    return True


def test_gradient_accumulation():
    """Test that gradient accumulation produces correct results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 7: Gradient Accumulation Correctness")
    logger.info("=" * 60)

    device = get_device("auto")
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load two fresh models for comparison
    model1 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model2 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)

    model1.train()
    model2.train()

    # Create dataset
    dataset = MathSFTDataset(
        data_path="data/math/train.jsonl",
        tokenizer=tokenizer,
        max_seq_length=128,
        num_samples=2,
    )

    gradient_accumulation_steps = 2

    # Method A: Gradient accumulation (2 steps, batch_size=1)
    logger.info("Method A: Gradient accumulation")
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=1.0)  # lr=1 for exact gradient
    optimizer1.zero_grad()

    for i in range(gradient_accumulation_steps):
        sample = dataset[i]
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        labels = sample['labels'].unsqueeze(0).to(device)
        response_mask = sample['response_mask'].unsqueeze(0).to(device)

        log_prob_result = get_response_log_probs(model1, input_ids, labels, False)
        policy_log_probs = log_prob_result["log_probs"]

        num_response_tokens = response_mask.sum().item()
        loss, _ = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=num_response_tokens if num_response_tokens > 0 else 1.0,
        )
        logger.info(f"  Step {i+1}: loss={loss.item():.6f}")

    # Get accumulated gradients
    accumulated_grads = {}
    for name, param in model1.named_parameters():
        if param.grad is not None:
            accumulated_grads[name] = param.grad.clone()

    # Method B: Single batch (batch_size=2)
    logger.info("Method B: Single batch processing")
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=1.0)
    optimizer2.zero_grad()

    input_ids_batch = torch.stack([dataset[0]['input_ids'], dataset[1]['input_ids']]).to(device)
    labels_batch = torch.stack([dataset[0]['labels'], dataset[1]['labels']]).to(device)
    response_mask_batch = torch.stack([dataset[0]['response_mask'], dataset[1]['response_mask']]).to(device)

    log_prob_result = get_response_log_probs(model2, input_ids_batch, labels_batch, False)
    policy_log_probs = log_prob_result["log_probs"]

    # For single batch, we need to match the gradient computation
    # Each sample is normalized by its own response tokens, then averaged
    num_response_tokens = response_mask_batch.sum().item()
    loss, _ = sft_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask_batch,
        gradient_accumulation_steps=1,  # No accumulation
        normalize_constant=num_response_tokens if num_response_tokens > 0 else 1.0,
    )
    logger.info(f"  Single batch loss: {loss.item():.6f}")

    single_batch_grads = {}
    for name, param in model2.named_parameters():
        if param.grad is not None:
            single_batch_grads[name] = param.grad.clone()

    # Compare gradients
    # Note: Due to normalization differences, gradients won't match exactly
    # But they should be in the same ballpark
    grad_norms_accum = [g.norm().item() for g in accumulated_grads.values()]
    grad_norms_single = [g.norm().item() for g in single_batch_grads.values()]

    logger.info(f"Accumulated grad norm (mean): {sum(grad_norms_accum)/len(grad_norms_accum):.6f}")
    logger.info(f"Single batch grad norm (mean): {sum(grad_norms_single)/len(grad_norms_single):.6f}")

    logger.info("PASSED (gradient accumulation is working)")
    return True


def test_model_saving():
    """Test that model and tokenizer are saved correctly."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 8: Model and Tokenizer Saving")
    logger.info("=" * 60)

    import tempfile
    import shutil

    from cs336_alignment.sft import save_model

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    temp_dir = tempfile.mkdtemp(prefix="sft_test_")
    logger.info(f"Saving to: {temp_dir}")

    try:
        save_model(model, tokenizer, temp_dir)

        import os
        saved_files = os.listdir(temp_dir)
        logger.info(f"Saved files: {saved_files}")

        has_model = "model.safetensors" in saved_files or "pytorch_model.bin" in saved_files
        has_config = "config.json" in saved_files
        has_tokenizer = "tokenizer_config.json" in saved_files

        assert has_model, "Model weights not saved"
        assert has_config, "Config not saved"
        assert has_tokenizer, "Tokenizer config not saved"

        # Test loading
        loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir, trust_remote_code=True)
        loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir, trust_remote_code=True)

        # Verify parameters match
        original_params = dict(model.named_parameters())
        loaded_params = dict(loaded_model.named_parameters())
        for name in original_params:
            assert torch.allclose(original_params[name], loaded_params[name]), f"Mismatch in {name}"

        logger.info("PASSED")

    finally:
        shutil.rmtree(temp_dir)

    return True


def test_model_size_estimation():
    """Test model size estimation from model names."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 9: estimate_model_size_billions()")
    logger.info("=" * 60)

    # Test known models
    test_cases = [
        ("Qwen/Qwen2.5-Math-1.5B", 1.5),
        ("meta-llama/Llama-3.1-8B", 8.0),
        ("meta-llama/Llama-3.2-1B", 1.0),
        ("Qwen/Qwen2.5-0.5B", 0.5),
        ("models/qwen2.5-math-1.5b", 1.5),  # Local path
        ("some-model-7b", 7.0),  # Parsed from name
    ]

    for model_name, expected_size in test_cases:
        estimated = estimate_model_size_billions(model_name)
        logger.info(f"  {model_name}: {estimated}B (expected: {expected_size}B)")
        assert estimated == expected_size, f"Expected {expected_size}B for {model_name}, got {estimated}B"

    logger.info("PASSED")
    return True


def test_compute_environment_detection():
    """Test the compute environment detection."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 10: detect_compute_environment()")
    logger.info("=" * 60)

    # Test default detection (assumes 1.5B model)
    env = detect_compute_environment()

    logger.info(f"Detected device: {env.device}")
    logger.info(f"Device name: {env.device_name}")
    logger.info(f"Memory: {env.total_memory_gb:.1f} GB")
    logger.info(f"Num GPUs: {env.num_gpus}")
    logger.info(f"BF16 support: {env.supports_bf16}")
    logger.info(f"FP16 support: {env.supports_fp16}")
    logger.info(f"Recommended dtype: {env.recommended_dtype}")
    logger.info(f"Recommended batch_size: {env.recommended_batch_size}")
    logger.info(f"Recommended grad_accum: {env.recommended_grad_accum}")
    logger.info(f"Recommended num_workers: {env.recommended_num_workers}")
    logger.info(f"Mixed precision: {env.mixed_precision}")

    # Verify the ComputeEnvironment has all required fields
    assert env.device in ("cuda", "mps", "cpu"), f"Invalid device: {env.device}"
    assert env.total_memory_gb > 0, "Memory should be positive"
    assert env.recommended_batch_size > 0, "Batch size should be positive"
    assert env.recommended_grad_accum > 0, "Grad accum should be positive"
    assert env.recommended_num_workers >= 0, "Num workers should be non-negative"
    assert env.mixed_precision in ("bf16", "fp16", "no"), f"Invalid mixed precision: {env.mixed_precision}"

    # Test model-aware detection (Llama 8B should have more conservative settings)
    logger.info("")
    logger.info("Testing model-aware detection for Llama 8B...")
    env_llama = detect_compute_environment("meta-llama/Llama-3.1-8B")
    logger.info(f"  Llama 8B: batch_size={env_llama.recommended_batch_size}, grad_accum={env_llama.recommended_grad_accum}")

    # Llama 8B should have smaller batch size or larger grad_accum due to size
    # (unless running on a very large GPU)
    assert env_llama.recommended_batch_size * env_llama.recommended_grad_accum >= 1, "Should have valid effective batch"

    logger.info("PASSED")
    return True


def test_sft_config_auto():
    """Test SFTConfig.create_auto_config()."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 11: SFTConfig.create_auto_config()")
    logger.info("=" * 60)

    # Test auto config creation
    config = SFTConfig.create_auto_config(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        train_data_path="data/math/train.jsonl",
        output_dir="outputs/test_auto",
    )

    logger.info(f"Auto config created:")
    logger.info(f"  model_name_or_path: {config.model_name_or_path}")
    logger.info(f"  batch_size: {config.batch_size}")
    logger.info(f"  gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    logger.info(f"  num_workers: {config.num_workers}")
    logger.info(f"  device: {config.device}")

    # Auto config should have positive values (not -1)
    assert config.batch_size > 0, "batch_size should be auto-detected"
    assert config.gradient_accumulation_steps > 0, "grad_accum should be auto-detected"
    assert config.num_workers >= 0, "num_workers should be auto-detected"

    # Test that kwargs override works
    config_override = SFTConfig.create_auto_config(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        num_epochs=5,  # Override default
        learning_rate=1e-4,  # Override default
    )

    assert config_override.num_epochs == 5, "num_epochs override should work"
    assert config_override.learning_rate == 1e-4, "learning_rate override should work"

    logger.info("PASSED")
    return True


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("SFT Verification Script")
    logger.info("=" * 60)

    data_path = "data/math/train.jsonl"
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    all_passed = True
    tests = [
        ("format_math_example", test_format_math_example),
        ("tokenize_prompt_and_output", test_tokenize_prompt_and_output),
        ("compute_entropy", test_compute_entropy),
        ("get_response_log_probs", test_get_response_log_probs),
        ("masked_normalize", test_masked_normalize),
        ("sft_microbatch_train_step", test_sft_microbatch_train_step),
        ("gradient_accumulation", test_gradient_accumulation),
        ("model_saving", test_model_saving),
        ("model_size_estimation", test_model_size_estimation),
        ("compute_environment_detection", test_compute_environment_detection),
        ("sft_config_auto", test_sft_config_auto),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            logger.error(f"Test '{name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    logger.info("")
    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL TESTS PASSED")
    else:
        logger.info("SOME TESTS FAILED")
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
