"""CIFAR-10 training for a small ViT, comparing precision schemes.

Setup:
  - CIFAR-10 subset preprocessed to /tmp/cifar_small.pt (10k train, 5k test)
  - Small ViT: dim=96, depth=4, heads=4, FFN hidden=192 (2x), patch=4
  - AccumAdam optimizer, lr=3e-3, cosine schedule, 2000 steps, batch=128
  - Runs across 4 precision configurations:
      1. fp32 master / no quant        (reference)
      2. bf16+int8 master / no quant   (accumulator alone)
      3. bf16+int8 master / int4       (accumulator + int4 forward)
      4. naive bf16 master / int4      (lossy baseline)
"""

import math
import time
import torch
import torch.nn.functional as F

from vit_peer import PeerViT, sync_all, count_params
from optim import AccumAdam


def load_cifar(path="/tmp/cifar_small.pt"):
    d = torch.load(path, weights_only=True)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip + random 4-pixel crop (standard CIFAR aug)."""
    B, C, H, W = x.shape
    # Horizontal flip
    flip = torch.rand(B) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(-1)
    # Random crop with 4-pixel reflect padding
    pad = 4
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = torch.empty_like(x)
    for i in range(B):
        top = torch.randint(0, 2 * pad + 1, (1,)).item()
        left = torch.randint(0, 2 * pad + 1, (1,)).item()
        out[i] = x_pad[i, :, top : top + H, left : left + W]
    return out


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 100) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, x, y, batch_size=256):
    model.eval()
    sync_all(model)
    correct = 0
    total_loss = 0.0
    n = x.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb, yb = x[i : i + batch_size], y[i : i + batch_size]
            logits = model(xb)
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(-1) == yb).sum().item()
    model.train()
    return correct / n, total_loss / n


def train_cifar(master, quant, seed=1, n_steps=1500, batch_size=128,
                base_lr=3e-3, eval_every=250):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_cifar()
    n_train = x_tr.shape[0]

    model = PeerViT(
        image_size=32, patch_size=4, num_classes=10,
        dim=96, depth=4, num_heads=4,
        num_experts=0, top_k=0,  # unused since mlp_type='ffn'
        quant=quant, master=master,
        mlp_type="ffn", mlp_hidden_mult=2,
    )
    opt = AccumAdam(model, lr=base_lr)

    log = {"step": [], "train_loss": [], "val_loss": [], "val_acc": []}
    loss_ema = None
    t0 = time.time()

    for step in range(n_steps):
        opt.lr = cosine_lr(step, n_steps, base_lr)

        idx = torch.randint(0, n_train, (batch_size,))
        xb = augment_batch(x_tr[idx])
        yb = y_tr[idx]

        sync_all(model)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

        l = loss.item()
        loss_ema = l if loss_ema is None else 0.97 * loss_ema + 0.03 * l

        if (step + 1) % eval_every == 0 or step == 0:
            val_acc, val_loss = evaluate(model, x_te, y_te)
            log["step"].append(step + 1)
            log["train_loss"].append(loss_ema)
            log["val_loss"].append(val_loss)
            log["val_acc"].append(val_acc)
            print(f"    step {step+1:>4}/{n_steps}  "
                  f"lr={opt.lr:.4f}  "
                  f"train_ema={loss_ema:.3f}  "
                  f"val_loss={val_loss:.3f}  val_acc={val_acc:.3f}  "
                  f"({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    return {"log": log, "final_val_acc": log["val_acc"][-1],
            "final_val_loss": log["val_loss"][-1],
            "elapsed": elapsed}


def main():
    configs = [
        ("fp32 master / no quant      ", "fp32", "none"),
        ("bf16+int8 master / no quant ", "accum", "none"),
        ("bf16+int8 master / int4     ", "accum", "int4"),
        ("naive bf16 master / int4    ", "naive_bf16", "int4"),
    ]

    print("=" * 78)
    print("ViT on CIFAR-10: precision scheme comparison")
    print("=" * 78)
    ref = PeerViT(
        image_size=32, patch_size=4, num_classes=10,
        dim=96, depth=4, num_heads=4,
        num_experts=0, top_k=0,
        mlp_type="ffn", mlp_hidden_mult=2,
    )
    p = count_params(ref)
    print(f"  Model: dim=96, depth=4, heads=4, FFN hidden=192 (2x), patch=4")
    print(f"  AccumLinear weights: {p['accum_weights']:>7,}")
    print(f"  Other params       : {p['accum_biases'] + p['other']:>7,}")
    print(f"  Total              : {p['total']:>7,}")
    print(f"  Training: 10k CIFAR-10 subset, 5k test, batch=128, 1500 steps,")
    print(f"            AccumAdam lr=3e-3 cosine, aug=flip+4pix crop")
    print()
    del ref

    results = {}
    for name, master, quant in configs:
        print(f"=== {name} (master={master}, quant={quant}) ===")
        r = train_cifar(master=master, quant=quant, seed=1)
        results[name] = r
        print(f"    final val_acc={r['final_val_acc']:.4f}  "
              f"val_loss={r['final_val_loss']:.3f}  "
              f"elapsed={r['elapsed']:.0f}s")
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'config':<32} {'val_acc':>10} {'val_loss':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*10}")
    for name, _, _ in configs:
        r = results[name]
        print(f"  {name:<32} {r['final_val_acc']:>10.4f} {r['final_val_loss']:>10.3f}")

    # Learning curves at fixed checkpoints
    print()
    print("Learning curves (val_acc at each eval):")
    steps = results[configs[0][0]]["log"]["step"]
    header = "  step " + "".join(f"{n[:20]:>22}" for n, _, _ in configs)
    print(header)
    for i, s in enumerate(steps):
        row = f"  {s:>4} "
        for name, _, _ in configs:
            row += f"{results[name]['log']['val_acc'][i]:>22.4f}"
        print(row)


if __name__ == "__main__":
    main()
