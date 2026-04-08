"""Downstream task evaluation via lm-evaluation-harness."""

from typing import Optional


def run_downstream_eval(
    model,
    tokenizer,
    tasks: list[str],
    num_fewshot: int = 0,
    batch_size: str | int = "auto",
    device: str = "cuda",
    limit: Optional[int] = None,
) -> dict:
    """Run downstream evaluation tasks using lm-eval harness.

    Args:
        model: The language model to evaluate (HF-compatible).
        tokenizer: Tokenizer for the model.
        tasks: List of task names (e.g., ['lambada_openai', 'hellaswag']).
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size for evaluation ('auto' for adaptive).
        device: Device string.
        limit: Optional limit on number of examples per task.

    Returns:
        Dict mapping task_name -> {metric_name: value}.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
    )

    # Extract per-task metrics
    parsed = {}
    for task_name in tasks:
        task_results = results["results"].get(task_name, {})
        parsed[task_name] = {
            k: v for k, v in task_results.items() if not k.startswith("_")
        }

    return parsed


def compute_quality_delta(
    baseline_results: dict, current_results: dict
) -> dict:
    """Compute per-task quality delta between baseline and current results.

    Args:
        baseline_results: Results from Step 0.
        current_results: Results from current step.

    Returns:
        Dict mapping task_name -> {metric_name: delta_value}.
        Positive delta means current is better; negative means regression.
    """
    deltas = {}
    for task in baseline_results:
        if task not in current_results:
            continue
        deltas[task] = {}
        for metric in baseline_results[task]:
            if metric in current_results[task]:
                base_val = baseline_results[task][metric]
                curr_val = current_results[task][metric]
                if isinstance(base_val, (int, float)) and isinstance(curr_val, (int, float)):
                    deltas[task][metric] = curr_val - base_val
    return deltas


def format_results_table(
    results: dict,
    baseline: dict | None = None,
) -> str:
    """Format evaluation results as a markdown table.

    Args:
        results: Current evaluation results.
        baseline: Optional baseline results for delta computation.

    Returns:
        Markdown-formatted table string.
    """
    lines = ["| Task | Metric | Value |" + (" Delta |" if baseline else "")]
    lines.append("|------|--------|-------|" + ("-------|" if baseline else ""))

    for task in sorted(results.keys()):
        for metric in sorted(results[task].keys()):
            val = results[task][metric]
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)

            row = f"| {task} | {metric} | {val_str} |"
            if baseline and task in baseline and metric in baseline[task]:
                base_val = baseline[task][metric]
                if isinstance(base_val, (int, float)) and isinstance(val, (int, float)):
                    delta = val - base_val
                    row += f" {delta:+.4f} |"
                else:
                    row += " - |"
            lines.append(row)

    return "\n".join(lines)
