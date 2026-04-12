"""Training harness comparing precision schemes on a PEER-ViT.

We train the same tiny PEER-ViT on a synthetic classification task under
several (master, quant) configurations and compare training dynamics.

Baselines:
  - fp32 master, no quant                : upper bound
  - fp32 master, int4 forward quant      : isolates int4 quant effect
  - bf16+int8 master, int4 forward quant : the main proposal
  - bf16+int8 master, no quant           : isolates accumulator effect
  - naive bf16 master, int4 forward quant: lossy baseline
"""

import time
import torch
import torch.nn.functional as F

from vit_peer import PeerViT, sync_all, count_params
from optim import AccumAdam


def make_data(
    n_samples: int,
    image_size: int,
    num_classes: int,
    seed: int,
    teacher_seed: int = 999,
):
    """Synthetic classification: labels = argmax of a fixed linear teacher.

    The teacher is shared across train/val (via teacher_seed), so the task is
    well-defined and val accuracy is meaningful. Linear classification over
    random Gaussian inputs is learnable to high accuracy with a ViT.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 3, image_size, image_size, generator=g)
    flat = x.view(n_samples, -1)
    d_in = flat.shape[1]
    gt = torch.Generator().manual_seed(teacher_seed)
    W = torch.randn(d_in, num_classes, generator=gt) / d_in ** 0.5
    y = (flat @ W).argmax(dim=-1)
    return x, y


def train_run(
    master: str,
    quant: str,
    n_train: int = 1024,
    n_val: int = 512,
    n_steps: int = 800,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
    image_size: int = 16,
    num_classes: int = 10,
    num_experts: int = 64,
    top_k: int = 64,  # dense PEER: all experts active (avoids routing-collapse)
    dim: int = 48,
    depth: int = 2,
    memorization: bool = False,
):
    torch.manual_seed(seed)
    x_train, y_train = make_data(
        n_train, image_size, num_classes, seed=100 + seed
    )
    if memorization:
        x_val, y_val = x_train, y_train
    else:
        x_val, y_val = make_data(
            n_val, image_size, num_classes, seed=200 + seed
        )

    model = PeerViT(
        image_size=image_size,
        patch_size=4,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        num_heads=4,
        num_experts=num_experts,
        top_k=top_k,
        quant=quant,
        master=master,
    )
    opt = AccumAdam(model, lr=lr)

    losses = []
    val_acc_curve = []
    log_every = max(1, n_steps // 10)
    t0 = time.time()

    for step in range(n_steps):
        idx = torch.randint(0, n_train, (batch_size,))
        xb, yb = x_train[idx], y_train[idx]

        sync_all(model)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

        losses.append(loss.item())

        if (step + 1) % log_every == 0 or step == 0:
            with torch.no_grad():
                sync_all(model)
                model.eval()
                val_logits = model(x_val)
                val_acc = (val_logits.argmax(-1) == y_val).float().mean().item()
                model.train()
            val_acc_curve.append((step + 1, val_acc))

    elapsed = time.time() - t0
    return {
        "losses": losses,
        "val_acc_curve": val_acc_curve,
        "final_loss": sum(losses[-20:]) / 20,
        "final_val_acc": val_acc_curve[-1][1] if val_acc_curve else 0.0,
        "elapsed": elapsed,
        "n_params": count_params(model),
    }


def main():
    configs = [
        ("fp32 master / no quant       ", "fp32", "none"),
        ("fp32 master / int4 forward   ", "fp32", "int4"),
        ("bf16+int8 master / int4      ", "accum", "int4"),
        ("bf16+int8 master / no quant  ", "accum", "none"),
        ("naive bf16 master / int4     ", "naive_bf16", "int4"),
        ("naive bf16 master / no quant ", "naive_bf16", "none"),
    ]
    seeds = [1, 2, 3]

    print("=" * 78)
    print("PEER-ViT training: precision scheme comparison")
    print("=" * 78)
    model = PeerViT(
        image_size=16, patch_size=4, num_classes=10,
        dim=48, depth=2, num_heads=4, num_experts=64, top_k=64,
    )
    p = count_params(model)
    print(f"  Model: dim=48, depth=2, num_heads=4, num_experts=64, top_k=64 (dense)")
    print(f"  AccumLinear weights: {p['accum_weights']:,} (bf16+int8+int4 target)")
    print(f"  Other params       : {p['accum_biases']+p['other']:,} (fp32)")
    print(f"  Total              : {p['total']:,}")
    print(f"  Task: linear-teacher classification, 10 classes, 16x16 images")
    print(f"  Training: 1024 samples, 800 steps, batch=32, lr=3e-3, AccumAdam")
    print(f"  Seeds: {seeds}")
    print()
    del model

    # results[name] = list of per-seed result dicts
    results = {name: [] for name, _, _ in configs}

    for name, master, quant in configs:
        print(f"=== {name} (master={master}, quant={quant}) ===")
        for seed in seeds:
            r = train_run(master=master, quant=quant, seed=seed)
            results[name].append(r)
            print(f"    seed={seed}: "
                  f"loss={r['final_loss']:.4f}  "
                  f"val_acc={r['final_val_acc']:.4f}  "
                  f"({r['elapsed']:.1f}s)  "
                  f"loss@50/100/200/400="
                  f"{r['losses'][49]:.2f}/{r['losses'][99]:.2f}/"
                  f"{r['losses'][199]:.2f}/{r['losses'][399]:.2f}")
        print()

    print("=" * 78)
    print("SUMMARY (mean ± std over seeds)")
    print("=" * 78)
    print(f"  {'config':<34} {'loss':>18} {'val_acc':>18}")
    print(f"  {'-'*34} {'-'*18} {'-'*18}")

    def stats(vals):
        n = len(vals)
        m = sum(vals) / n
        s = (sum((v - m) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
        return m, s

    for name, _, _ in configs:
        rs = results[name]
        lm, ls = stats([r["final_loss"] for r in rs])
        vm, vs = stats([r["final_val_acc"] for r in rs])
        print(f"  {name:<34} {lm:>8.4f} ± {ls:<6.4f}  {vm:>8.4f} ± {vs:<6.4f}")

    # Also report loss curve alignment of bf16+int8 vs fp32
    print()
    print("Training-loss curves (seed=1):")
    print(f"  {'step':>6}  {'fp32':>10}  {'bf16+int8+int4':>16}  "
          f"{'naive_bf16+int4':>16}")
    fp32_losses = results["fp32 master / no quant       "][0]["losses"]
    accum_losses = results["bf16+int8 master / int4      "][0]["losses"]
    naive_losses = results["naive bf16 master / int4     "][0]["losses"]
    n_log_steps = len(fp32_losses)
    for step in [1, 25, 50, 100, 200, 400, 600, n_log_steps]:
        i = min(step - 1, n_log_steps - 1)
        print(f"  {step:>6}  {fp32_losses[i]:>10.4f}  "
              f"{accum_losses[i]:>16.4f}  {naive_losses[i]:>16.4f}")

    # ---------------------------------------------------------------------
    # Memorization test: train == val (256 samples). Pure gradient-fidelity
    # check — no generalization required. A working scheme should drive
    # train loss close to 0.
    # ---------------------------------------------------------------------
    print()
    print("=" * 78)
    print("MEMORIZATION TEST (train==val, 256 samples, 400 steps)")
    print("Tests whether the model can fit the training set — pure gradient test.")
    print("=" * 78)
    mem_results = {}
    for name, master, quant in configs:
        r = train_run(
            master=master,
            quant=quant,
            n_train=256,
            n_val=256,
            n_steps=400,
            batch_size=32,
            lr=3e-3,
            seed=1,
            memorization=True,
        )
        mem_results[name] = r
        print(f"  {name} final_loss={r['final_loss']:.4f}  "
              f"memorize_acc={r['final_val_acc']:.4f}  "
              f"loss@50/100/200/400="
              f"{r['losses'][49]:.2f}/{r['losses'][99]:.2f}/"
              f"{r['losses'][199]:.2f}/{r['losses'][-1]:.2f}")

    # ---------------------------------------------------------------------
    # LOW-LR stress test: Adam updates are ~lr in magnitude, so lr=3e-5
    # puts updates close to the bf16 ULP scale at typical weight magnitudes.
    # This is where naive bf16 should *really* struggle — each update is near
    # the rounding threshold — while the accumulator captures sub-ULP info.
    # ---------------------------------------------------------------------
    print()
    print("=" * 78)
    print("LOW-LR STRESS TEST (memorization, lr=3e-5, 800 steps)")
    print("Small updates expose bf16 rounding error. Accumulator should shine here.")
    print("=" * 78)
    stress_results = {}
    for name, master, quant in configs:
        r = train_run(
            master=master,
            quant=quant,
            n_train=256,
            n_val=256,
            n_steps=800,
            batch_size=32,
            lr=3e-5,
            seed=1,
            memorization=True,
        )
        stress_results[name] = r
        print(f"  {name} final_loss={r['final_loss']:.4f}  "
              f"acc={r['final_val_acc']:.4f}  "
              f"loss@100/400/800="
              f"{r['losses'][99]:.3f}/{r['losses'][399]:.3f}/{r['losses'][-1]:.3f}")

    print()
    print("Stress-test summary (lower = better tracking of fp32 ideal):")
    print(f"  {'config':<34} {'final loss':>12}")
    for name, _, _ in configs:
        print(f"  {name:<34} {stress_results[name]['final_loss']:>12.4f}")


if __name__ == "__main__":
    main()
