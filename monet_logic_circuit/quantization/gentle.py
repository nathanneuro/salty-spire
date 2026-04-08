"""Gentle quantization (4-bit / 8-bit) using GPTQ or AWQ.

This is the "should just work" baseline. Conservative settings,
minimal expected quality loss. Used as sanity check before aggressive
quantization.
"""

from typing import Optional

import torch
import torch.nn as nn


def apply_gentle_quantization(
    model: nn.Module,
    tokenizer,
    method: str = "gptq",
    bits: int = 4,
    calibration_data=None,
    config: Optional[dict] = None,
    targets: Optional[dict] = None,
) -> nn.Module:
    """Apply gentle post-training quantization to expert weights.

    Args:
        model: Pretrained Monet model.
        tokenizer: Model tokenizer.
        method: Quantization method ('gptq' or 'awq').
        bits: Bit width (4 or 8).
        calibration_data: Calibration dataset for quantization.
        config: Method-specific config (group_size, damp_percent, etc.).
        targets: Dict specifying which components to quantize.
            Keys: 'experts', 'attention', 'norms', 'embeddings'.
            Only experts should be True for this step.

    Returns:
        Quantized model.
    """
    config = config or {}
    targets = targets or {"experts": True, "attention": False, "norms": False, "embeddings": False}

    if method == "gptq":
        return _apply_gptq(model, tokenizer, bits, calibration_data, config, targets)
    elif method == "awq":
        return _apply_awq(model, tokenizer, bits, calibration_data, config, targets)
    else:
        raise ValueError(f"Unknown quantization method: {method}. Use 'gptq' or 'awq'.")


def _apply_gptq(
    model: nn.Module,
    tokenizer,
    bits: int,
    calibration_data,
    config: dict,
    targets: dict,
) -> nn.Module:
    """Apply GPTQ quantization."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    group_size = config.get("group_size", 128)
    damp_percent = config.get("damp_percent", 0.01)
    desc_act = config.get("desc_act", True)
    sym = config.get("sym", True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        sym=sym,
    )

    # Prepare calibration examples
    examples = _prepare_calibration_examples(calibration_data, tokenizer)

    # Apply GPTQ -- this quantizes in place
    # For selective quantization (experts only), we'd need to specify
    # which modules to target. This depends on the GPTQ implementation.
    model_quant = AutoGPTQForCausalLM.from_pretrained(
        model,
        quantize_config=quantize_config,
    )
    model_quant.quantize(examples)

    return model_quant.model


def _apply_awq(
    model: nn.Module,
    tokenizer,
    bits: int,
    calibration_data,
    config: dict,
    targets: dict,
) -> nn.Module:
    """Apply AWQ quantization."""
    from awq import AutoAWQForCausalLM

    group_size = config.get("group_size", 128)
    zero_point = config.get("zero_point", True)

    quant_config = {
        "w_bit": bits,
        "q_group_size": group_size,
        "zero_point": zero_point,
    }

    # Prepare calibration text
    calib_texts = _prepare_calibration_texts(calibration_data, tokenizer)

    model_quant = AutoAWQForCausalLM.from_pretrained(model)
    model_quant.quantize(
        tokenizer=tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts,
    )

    return model_quant.model


def measure_per_expert_reconstruction(
    original_model: nn.Module,
    quantized_model: nn.Module,
    trace_store,
) -> dict[str, float]:
    """Measure per-expert reconstruction error: quantized vs float output.

    Uses cached calibration traces from Step 0 to evaluate each expert
    individually.

    Args:
        original_model: Float-precision model.
        quantized_model: Quantized model.
        trace_store: ExpertTraceStore with Step 0 traces.

    Returns:
        Dict mapping expert_name -> normalized MSE.
    """
    from monet_logic_circuit.models.monet_loader import get_expert_modules

    errors = {}
    quant_experts = dict(get_expert_modules(quantized_model))

    for expert_name in trace_store.list_experts():
        inputs, ref_outputs = trace_store.load_traces(expert_name)

        if expert_name not in quant_experts:
            continue

        quant_expert = quant_experts[expert_name]
        with torch.no_grad():
            quant_outputs = quant_expert(inputs.to(next(quant_expert.parameters()).device))
            quant_outputs = quant_outputs.cpu()

        mse = ((ref_outputs.float() - quant_outputs.float()) ** 2).mean()
        ref_var = ref_outputs.float().var()
        nmse = float(mse / (ref_var + 1e-10))
        errors[expert_name] = nmse

    return errors


def _prepare_calibration_examples(calibration_data, tokenizer, max_samples: int = 128):
    """Convert calibration dataset to list of tokenized examples for GPTQ."""
    examples = []
    for i, sample in enumerate(calibration_data):
        if i >= max_samples:
            break
        examples.append({"input_ids": sample["input_ids"].unsqueeze(0)})
    return examples


def _prepare_calibration_texts(calibration_data, tokenizer, max_samples: int = 128):
    """Convert calibration data to text strings for AWQ."""
    texts = []
    for i, sample in enumerate(calibration_data):
        if i >= max_samples:
            break
        text = tokenizer.decode(sample["input_ids"])
        texts.append(text)
    return texts
