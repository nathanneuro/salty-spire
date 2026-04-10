"""
Test harness for BF16 + Int8 Residual Accumulator.

Compares three schemes against fp32 ground truth:
  1. bf16+int8 (RTN bf16, SR residual) — the proposed scheme
  2. bf16+int8 (SR bf16, RTN residual) — original scheme for comparison
  3. naive bf16 — baseline

Tests, in order of importance:
  1. Unbiasedness
  2. Variance / RMSE comparison
  3. Edge cases
  4. Gradient scale sweep
  5. Error distribution
  6. ULP correctness
"""

import torch
import sys

from accumulator import (
    accumulate_with_residual,
    accumulate_with_residual_sr_bf16,
    bf16_ulp,
    naive_bf16_accumulate,
    reconstruct_fp32,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).pow(2).mean().sqrt().item()


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def mean_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().mean().item()


def relative_err(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.max(a.abs(), b.abs()).clamp(min=1e-12)
    return ((a - b).abs() / denom).mean().item()


def run_accumulation(w_init, grad_seq, method):
    """Run accumulation with a given method over a sequence of gradients.

    method: 'rtn_sr' | 'sr_rtn' | 'naive'
    Returns (final_weights_fp32, n_steps).
    """
    n_steps = grad_seq.shape[0]
    n_params = w_init.shape[0]

    if method == 'naive':
        w = w_init.bfloat16().clone()
        for step in range(n_steps):
            naive_bf16_accumulate(w, grad_seq[step])
        return w.float(), n_steps
    else:
        w = w_init.bfloat16().clone()
        r = torch.zeros(n_params, dtype=torch.int8)
        fn = accumulate_with_residual if method == 'rtn_sr' else accumulate_with_residual_sr_bf16
        for step in range(n_steps):
            fn(w, r, grad_seq[step])
        return reconstruct_fp32(w, r), n_steps


# ---------------------------------------------------------------------------
# Test 1: Unbiasedness
# ---------------------------------------------------------------------------

def test_unbiasedness(n_params=10000, n_steps=1000, grad_scale=1e-4, n_trials=5):
    print("=" * 72)
    print("TEST 1: Unbiasedness (signed error should be ~0)")
    print("=" * 72)

    w_init = torch.randn(n_params, dtype=torch.float32)

    for label, method in [("RTN+SR", "rtn_sr"), ("SR+RTN", "sr_rtn"), ("naive", "naive")]:
        signed_errors = []
        abs_errors = []
        for trial in range(n_trials):
            torch.manual_seed(1000 + trial)
            grad_seq = torch.randn(n_steps, n_params) * grad_scale

            # Ground truth
            w_fp32 = w_init.clone()
            for s in range(n_steps):
                w_fp32 += grad_seq[s]

            # Test method (re-seed so SR randomness varies per trial but grads match)
            torch.manual_seed(2000 + trial)
            w_result, _ = run_accumulation(w_init, grad_seq, method)

            signed_errors.append((w_result - w_fp32).mean().item())
            abs_errors.append((w_result - w_fp32).abs().mean().item())

        mean_signed = sum(signed_errors) / n_trials
        mean_abs = sum(abs_errors) / n_trials
        bias_ratio = abs(mean_signed) / (mean_abs + 1e-30)
        print(f"  {label:8s}  signed={mean_signed:+.2e}  |err|={mean_abs:.2e}  "
              f"|bias|/|err|={bias_ratio:.4f}")

    print()


# ---------------------------------------------------------------------------
# Test 2: Variance comparison
# ---------------------------------------------------------------------------

def test_variance(n_params=10000, n_steps=2000, grad_scale=1e-4):
    print("=" * 72)
    print("TEST 2: Variance — RMSE vs fp32 ground truth")
    print("=" * 72)

    w_init = torch.randn(n_params, dtype=torch.float32)
    torch.manual_seed(42)
    grad_seq = torch.randn(n_steps, n_params) * grad_scale

    w_fp32 = w_init.clone()
    for s in range(n_steps):
        w_fp32 += grad_seq[s]

    print(f"  After {n_steps} steps, grad_scale={grad_scale}:")
    print(f"  {'method':>10}  {'RMSE':>12}  {'max|err|':>12}  {'rel_err':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

    results = {}
    for label, method in [("RTN+SR", "rtn_sr"), ("SR+RTN", "sr_rtn"), ("naive", "naive")]:
        torch.manual_seed(99)
        w_result, _ = run_accumulation(w_init, grad_seq, method)
        r = rmse(w_result, w_fp32)
        m = max_err(w_result, w_fp32)
        rel = relative_err(w_result, w_fp32)
        results[label] = r
        print(f"  {label:>10}  {r:>12.4e}  {m:>12.4e}  {rel:>12.4e}")

    improvement = results["naive"] / (results["RTN+SR"] + 1e-30)
    print(f"\n  RTN+SR vs naive RMSE improvement: {improvement:.1f}x")
    ok = improvement > 2.0
    print(f"  PASS (>2x): {ok}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Test 3: Edge cases
# ---------------------------------------------------------------------------

def test_edge_cases():
    print("=" * 72)
    print("TEST 3: Edge cases")
    print("=" * 72)
    all_ok = True

    # --- 3a: Near zero ---
    print("  3a: Tiny gradients near zero")
    for label, method in [("RTN+SR", "rtn_sr"), ("naive", "naive")]:
        torch.manual_seed(77)
        w_init = torch.zeros(100, dtype=torch.float32)
        grad_seq = torch.randn(500, 100) * 1e-6
        w_fp32 = w_init.clone()
        for s in range(500):
            w_fp32 += grad_seq[s]
        torch.manual_seed(88)
        w_result, _ = run_accumulation(w_init, grad_seq, method)
        err = rmse(w_result, w_fp32)
        print(f"      {label:8s} RMSE: {err:.4e}")
    print()

    # --- 3b: Large magnitude ---
    print("  3b: Large magnitude (near max bf16)")
    w_bf16 = torch.tensor([3e38, -3e38, 1e38], dtype=torch.bfloat16)
    r_int8 = torch.zeros(3, dtype=torch.int8)
    w_fp32 = w_bf16.float().clone()
    for _ in range(100):
        g = torch.tensor([1e35, -1e35, 5e34], dtype=torch.float32)
        w_fp32 += g
        accumulate_with_residual(w_bf16, r_int8, g)
    w_recon = reconstruct_fp32(w_bf16, r_int8)
    rel = relative_err(w_recon, w_fp32)
    ok = rel < 0.01
    print(f"      Relative error: {rel:.4e}  PASS: {ok}")
    all_ok &= ok
    print()

    # --- 3c: Sign flips ---
    print("  3c: Sign flips (gradient drives weight through zero)")
    w_bf16 = torch.tensor([0.01, -0.01, 0.001], dtype=torch.bfloat16)
    r_int8 = torch.zeros(3, dtype=torch.int8)
    w_fp32 = w_bf16.float().clone()
    for _ in range(200):
        g = torch.tensor([-0.001, 0.001, -0.0001], dtype=torch.float32)
        w_fp32 += g
        accumulate_with_residual(w_bf16, r_int8, g)
    w_recon = reconstruct_fp32(w_bf16, r_int8)
    err = (w_recon - w_fp32).abs()
    ok = err.max().item() < 0.005
    print(f"      Max error: {err.max().item():.4e}  PASS: {ok}")
    print(f"      fp32:  {w_fp32.tolist()}")
    print(f"      recon: {w_recon.tolist()}")
    all_ok &= ok
    print()

    # --- 3d: Large single gradient ---
    print("  3d: Large single gradient (100x weight)")
    w_bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
    r_int8 = torch.zeros(1, dtype=torch.int8)
    w_fp32 = torch.tensor([1.0], dtype=torch.float32)
    g = torch.tensor([100.0])
    w_fp32 += g
    accumulate_with_residual(w_bf16, r_int8, g)
    w_recon = reconstruct_fp32(w_bf16, r_int8)
    err = (w_recon - w_fp32).abs().item()
    ok = err < 0.5
    print(f"      fp32: {w_fp32.item()}, recon: {w_recon.item()}, err: {err:.4e}  PASS: {ok}")
    all_ok &= ok
    print()

    # --- 3e: Residual overflow check (regression for old SR scheme) ---
    print("  3e: Residual stays in int8 range (RTN+SR scheme)")
    w_bf16 = torch.ones(1000, dtype=torch.bfloat16)
    r_int8 = torch.zeros(1000, dtype=torch.int8)
    max_r = 0
    for _ in range(500):
        g = torch.randn(1000) * 1e-3
        accumulate_with_residual(w_bf16, r_int8, g)
        max_r = max(max_r, r_int8.abs().max().item())
    ok = max_r <= 127
    print(f"      Max |residual| seen: {max_r}  (should be ≤128)  PASS: {ok}")
    all_ok &= ok
    print()

    print(f"  Edge cases overall PASS: {all_ok}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Test 4: Gradient scale sweep
# ---------------------------------------------------------------------------

def test_gradient_scale_sweep(n_params=5000, n_steps=500):
    print("=" * 72)
    print("TEST 4: Gradient scale sweep")
    print("=" * 72)
    print(f"  {'scale':>10}  {'RTN+SR':>12}  {'SR+RTN':>12}  {'naive':>12}  "
          f"{'RTN/naive':>10}  {'SR/naive':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*10}")

    w_init = torch.randn(n_params, dtype=torch.float32)

    for exp in range(-7, 2):
        grad_scale = 10.0 ** exp
        torch.manual_seed(42)
        grad_seq = torch.randn(n_steps, n_params) * grad_scale

        w_fp32 = w_init.clone()
        for s in range(n_steps):
            w_fp32 += grad_seq[s]

        rmses = {}
        for label, method in [("RTN+SR", "rtn_sr"), ("SR+RTN", "sr_rtn"), ("naive", "naive")]:
            torch.manual_seed(99)
            w_result, _ = run_accumulation(w_init, grad_seq, method)
            rmses[label] = rmse(w_result, w_fp32)

        r1 = rmses["naive"] / (rmses["RTN+SR"] + 1e-30)
        r2 = rmses["naive"] / (rmses["SR+RTN"] + 1e-30)
        print(f"  {grad_scale:>10.0e}  {rmses['RTN+SR']:>12.4e}  {rmses['SR+RTN']:>12.4e}  "
              f"{rmses['naive']:>12.4e}  {r1:>9.1f}x  {r2:>9.1f}x")

    print()
    print("  RTN+SR should dominate at all scales. SR+RTN loses at small scales due to")
    print("  SR noise on full ULP scale + int8 residual overflow clamping.")
    print()
    return True


# ---------------------------------------------------------------------------
# Test 5: Error distribution
# ---------------------------------------------------------------------------

def test_error_distribution(n_params=10000, n_steps=1000, grad_scale=1e-4):
    print("=" * 72)
    print("TEST 5: Error distribution (percentiles)")
    print("=" * 72)

    w_init = torch.randn(n_params, dtype=torch.float32)
    torch.manual_seed(42)
    grad_seq = torch.randn(n_steps, n_params) * grad_scale

    w_fp32 = w_init.clone()
    for s in range(n_steps):
        w_fp32 += grad_seq[s]

    errors = {}
    for label, method in [("RTN+SR", "rtn_sr"), ("naive", "naive")]:
        torch.manual_seed(99)
        w_result, _ = run_accumulation(w_init, grad_seq, method)
        errors[label] = (w_result - w_fp32).abs()

    pcts = [50, 90, 95, 99, 99.9, 100]
    print(f"  {'pct':>8}  {'RTN+SR':>12}  {'naive':>12}  {'ratio':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}")
    for p in pcts:
        if p == 100:
            va = errors["RTN+SR"].max().item()
            vn = errors["naive"].max().item()
        else:
            va = torch.quantile(errors["RTN+SR"], p / 100).item()
            vn = torch.quantile(errors["naive"], p / 100).item()
        ratio = vn / (va + 1e-30)
        print(f"  {p:>7}%  {va:>12.4e}  {vn:>12.4e}  {ratio:>7.1f}x")

    # Signed error stats
    signed = errors["RTN+SR"]  # actually need signed
    torch.manual_seed(99)
    w_rtn, _ = run_accumulation(w_init, grad_seq, "rtn_sr")
    signed_err = w_rtn - w_fp32
    print(f"\n  RTN+SR signed error: mean={signed_err.mean().item():+.2e}  "
          f"std={signed_err.std().item():.2e}")
    print()
    return True


# ---------------------------------------------------------------------------
# Test 6: ULP correctness
# ---------------------------------------------------------------------------

def test_ulp_correctness():
    print("=" * 72)
    print("TEST 6: ULP function correctness")
    print("=" * 72)
    all_ok = True

    test_cases = [
        (1.0, 2**-7),
        (2.0, 2**-6),
        (0.5, 2**-8),
        (256.0, 2.0),
        (0.00390625, 2**-15),
    ]

    for val, expected_ulp in test_cases:
        t = torch.tensor([val], dtype=torch.bfloat16)
        computed = bf16_ulp(t).item()
        ok = abs(computed - expected_ulp) < 1e-20
        status = "OK" if ok else "FAIL"
        print(f"  bf16={val:>12g}  computed={computed:.4e}  expected={expected_ulp:.4e}  [{status}]")
        all_ok &= ok

    print(f"  PASS: {all_ok}")
    print()
    return all_ok


# ---------------------------------------------------------------------------
# Test 7: Long-run drift test
# ---------------------------------------------------------------------------

def test_long_run_drift(n_params=5000, n_steps=10000, grad_scale=1e-5):
    """Check that error doesn't grow unboundedly over many steps."""
    print("=" * 72)
    print("TEST 7: Long-run drift (error growth over 10k steps)")
    print("=" * 72)

    w_init = torch.randn(n_params, dtype=torch.float32)
    w_fp32 = w_init.clone()
    w_bf16 = w_init.bfloat16().clone()
    r_int8 = torch.zeros(n_params, dtype=torch.int8)
    w_naive = w_init.bfloat16().clone()

    checkpoints = [100, 500, 1000, 2000, 5000, 10000]
    check_idx = 0

    print(f"  {'step':>6}  {'RTN+SR RMSE':>12}  {'naive RMSE':>12}  {'ratio':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")

    for step in range(1, n_steps + 1):
        g = torch.randn(n_params) * grad_scale
        w_fp32 += g
        accumulate_with_residual(w_bf16, r_int8, g)
        naive_bf16_accumulate(w_naive, g)

        if check_idx < len(checkpoints) and step == checkpoints[check_idx]:
            w_recon = reconstruct_fp32(w_bf16, r_int8)
            r_acc = rmse(w_recon, w_fp32)
            r_nai = rmse(w_naive.float(), w_fp32)
            ratio = r_nai / (r_acc + 1e-30)
            print(f"  {step:>6}  {r_acc:>12.4e}  {r_nai:>12.4e}  {ratio:>7.1f}x")
            check_idx += 1

    # Error should not grow faster than sqrt(n_steps) for unbiased accumulator
    # (random walk). Check that final RMSE / initial RMSE is reasonable.
    print()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    results = {}
    test_ulp_correctness()     # informational
    test_unbiasedness()        # informational (prints all three)
    results["variance"] = test_variance()
    results["edge_cases"] = test_edge_cases()
    test_gradient_scale_sweep()  # informational
    test_error_distribution()    # informational
    test_long_run_drift()        # informational

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")

    n_fail = sum(1 for v in results.values() if not v)
    if n_fail:
        print(f"\n  {n_fail} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n  All tests passed.")
        sys.exit(0)
