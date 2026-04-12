"""CIFAR-10 training with *true sparse* PEER layers and APOLLO-mini optimizer.

Combines the full precision stack:
  - Sparse PEER (num_experts >> top_k, real gather/einsum, load-balancing loss)
  - APOLLO-mini optimizer (rank-1 projected Adam with tensor-wise scale)
  - bf16 + int8 residual master weight storage
  - int4 forward weight quantization via STE

Configurations compared:
  A) fp32 / Adam                               (reference)
  B) fp32 / APOLLO-mini                        (isolates APOLLO-mini effect)
  C) bf16+int8 / APOLLO-mini / int4 forward    (full stack)
  D) naive bf16 / APOLLO-mini / int4 forward   (no-residual baseline)
"""

import math
import time
import torch
import torch.nn.functional as F

from vit_peer import PeerViT, sync_all, count_params, SparsePEERLayer
from optim import AccumAdam
from optim_apollo import AccumApolloMini


def load_cifar(path="/tmp/cifar_small.pt"):
    d = torch.load(path, weights_only=True)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    flip = torch.rand(B) < 0.5
    x = x.clone()
    x[flip] = x[flip].flip(-1)
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


def make_model(master, quant):
    return PeerViT(
        image_size=32, patch_size=4, num_classes=10,
        dim=64, depth=2, num_heads=4,
        num_experts=128, top_k=8,  # 16:1 sparsity
        quant=quant, master=master,
        mlp_type="sparse_peer",
    )


def make_optimizer(kind, model, lr):
    if kind == "adam":
        return AccumAdam(model, lr=lr)
    elif kind == "apollo_mini":
        return AccumApolloMini(model, lr=lr, variant="mini")
    elif kind == "apollo_hybrid":
        return AccumApolloMini(model, lr=lr, variant="hybrid")
    raise ValueError(kind)


def train_one(master, quant, opt_kind, seed=1, n_steps=800, batch_size=64,
              base_lr=3e-3, eval_every=100, aux_weight=0.01):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_cifar()
    n_train = x_tr.shape[0]

    model = make_model(master, quant)
    opt = make_optimizer(opt_kind, model, base_lr)

    log = {"step": [], "train_loss": [], "aux_loss": [],
           "val_loss": [], "val_acc": []}
    loss_ema = None
    aux_ema = None
    t0 = time.time()

    for step in range(n_steps):
        opt.lr = cosine_lr(step, n_steps, base_lr)

        idx = torch.randint(0, n_train, (batch_size,))
        xb = augment_batch(x_tr[idx])
        yb = y_tr[idx]

        sync_all(model)
        logits = model(xb)
        aux = model.aux_loss()
        ce = F.cross_entropy(logits, yb)
        loss = ce + aux_weight * aux
        loss.backward()
        opt.step()

        ce_v = ce.item()
        aux_v = aux.item()
        loss_ema = ce_v if loss_ema is None else 0.97 * loss_ema + 0.03 * ce_v
        aux_ema = aux_v if aux_ema is None else 0.97 * aux_ema + 0.03 * aux_v

        if (step + 1) % eval_every == 0 or step == 0:
            val_acc, val_loss = evaluate(model, x_te, y_te)
            log["step"].append(step + 1)
            log["train_loss"].append(loss_ema)
            log["aux_loss"].append(aux_ema)
            log["val_loss"].append(val_loss)
            log["val_acc"].append(val_acc)
            print(f"    step {step+1:>4}/{n_steps}  "
                  f"lr={opt.lr:.4f}  "
                  f"ce={loss_ema:.3f}  "
                  f"aux={aux_ema:.3f}  "
                  f"val_loss={val_loss:.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"({time.time()-t0:.0f}s)")

    return {
        "log": log,
        "final_val_acc": log["val_acc"][-1],
        "final_val_loss": log["val_loss"][-1],
        "final_aux": log["aux_loss"][-1],
        "elapsed": time.time() - t0,
    }


def main():
    configs = [
        ("fp32 / Adam                       ", "fp32", "none", "adam"),
        ("fp32 / APOLLO-mini                ", "fp32", "none", "apollo_mini"),
        ("fp32 / APOLLO-hybrid              ", "fp32", "none", "apollo_hybrid"),
        ("bf16+int8 / APOLLO-hybrid / int4  ", "accum", "int4", "apollo_hybrid"),
        ("naive bf16 / APOLLO-hybrid / int4 ", "naive_bf16", "int4", "apollo_hybrid"),
    ]

    print("=" * 80)
    print("Sparse PEER ViT on CIFAR-10: full precision+optimizer stack test")
    print("=" * 80)
    ref = make_model("fp32", "none")
    p = count_params(ref)
    ref_adam = AccumAdam(ref, lr=1e-3)
    ref_mini = AccumApolloMini(ref, lr=1e-3, variant="mini")
    ref_hybrid = AccumApolloMini(ref, lr=1e-3, variant="hybrid")
    adam_bytes = sum(
        st["m"].numel() + st["v"].numel()
        for st in ref_adam.state.values()
    ) * 4
    mini_mem = ref_mini.memory_bytes()["total"]
    hybrid_mem = ref_hybrid.memory_bytes()["total"]

    print(f"  Model: dim=64, depth=2, heads=4, num_experts=128, top_k=8 (16:1 sparse)")
    print(f"  Total params        : {p['total']:>8,}")
    print(f"  AccumLinear weights : {p['accum_weights']:>8,} "
          f"(gets bf16+int8 + int4 treatment)")
    print(f"  Other params        : {p['accum_biases'] + p['other']:>8,}")
    print()
    print(f"  Optimizer state memory:")
    print(f"    Adam           : {adam_bytes / 1024:>8.1f} kB  (100%)")
    print(f"    APOLLO-mini    : {mini_mem / 1024:>8.1f} kB  "
          f"({mini_mem/adam_bytes*100:.1f}% of Adam)")
    print(f"    APOLLO-hybrid  : {hybrid_mem / 1024:>8.1f} kB  "
          f"({hybrid_mem/adam_bytes*100:.1f}% of Adam)")
    print()
    print(f"  Training: 10k CIFAR-10 subset, 800 steps, batch=64, lr=3e-3 cosine")
    print(f"  Load-balancing aux loss weight: 0.01")
    print()
    del ref, ref_adam, ref_mini, ref_hybrid

    results = {}
    for name, master, quant, opt_kind in configs:
        print(f"=== {name} (master={master}, quant={quant}, opt={opt_kind}) ===")
        r = train_one(master=master, quant=quant, opt_kind=opt_kind, seed=1)
        results[name] = r
        print(f"    final: val_acc={r['final_val_acc']:.4f}  "
              f"aux={r['final_aux']:.3f}  elapsed={r['elapsed']:.0f}s")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'config':<34} {'val_acc':>10} {'val_loss':>10} {'aux':>8}")
    print(f"  {'-'*34} {'-'*10} {'-'*10} {'-'*8}")
    for name, _, _, _ in configs:
        r = results[name]
        print(f"  {name:<34} {r['final_val_acc']:>10.4f} "
              f"{r['final_val_loss']:>10.3f} {r['final_aux']:>8.3f}")

    # Learning curve
    print()
    print("Val-accuracy curves:")
    steps = results[configs[0][0]]["log"]["step"]
    header = "  step " + "".join(f"{n[:16]:>18}" for n, _, _, _ in configs)
    print(header)
    for i, s in enumerate(steps):
        row = f"  {s:>4} "
        for name, _, _, _ in configs:
            row += f"{results[name]['log']['val_acc'][i]:>18.4f}"
        print(row)

    # Aux loss curve to show load balancing is working
    print()
    print("Aux (load-balancing) loss curves — lower = better routing balance:")
    header = "  step " + "".join(f"{n[:16]:>18}" for n, _, _, _ in configs)
    print(header)
    for i, s in enumerate(steps):
        row = f"  {s:>4} "
        for name, _, _, _ in configs:
            row += f"{results[name]['log']['aux_loss'][i]:>18.3f}"
        print(row)


if __name__ == "__main__":
    main()
