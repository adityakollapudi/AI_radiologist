"""Calibration: ECE, MCE, multiclass Brier score, reliability diagrams and temperature scaling (fit on VALIDATION only)."""
import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece_mce(prob, y, n_bins=15):
    conf = prob.max(1); correct = (prob.argmax(1) == np.asarray(y)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce, table = 0.0, 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            table.append({"bin": f"({lo:.2f},{hi:.2f}]", "n": int(m.sum()), "confidence": float(conf[m].mean()),
                          "accuracy": float(correct[m].mean()), "gap": float(gap)})
    return float(ece), float(mce), table


def brier(prob, y):
    oh = np.eye(prob.shape[1])[np.asarray(y)]
    return float(((prob - oh) ** 2).sum(1).mean())


def fit_temperature(val_logits, val_y, lo=0.05, hi=10.0):
    """Minimise validation NLL over a single scalar T (log-space LBFGS, then bounded)."""
    logits = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_y), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    t = float(np.clip(log_t.exp().item(), lo, hi))
    nll_before = float(F.cross_entropy(logits, y))
    nll_after = float(F.cross_entropy(logits / t, y))
    return t, {"val_nll_before": nll_before, "val_nll_after": nll_after, "hit_bound": t in (lo, hi)}


def calibration_report(logits, y, T, n_bins=15):
    out = {}
    for name, t in (("uncalibrated", 1.0), ("temperature_scaled", T)):
        p = softmax_np(logits, t)
        e, m, table = ece_mce(p, y, n_bins)
        out[name] = {"ece": e, "mce": m, "brier": brier(p, y), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                     "reliability_table": table}
    p90 = softmax_np(logits, T).max(1)
    sel = p90 >= 0.9
    out["accuracy_when_confidence_ge_0.9"] = float((softmax_np(logits, T).argmax(1)[sel] == np.asarray(y)[sel]).mean()) if sel.any() else None
    out["share_with_confidence_ge_0.9"] = float(sel.mean())
    out["temperature"] = float(T)
    return out


def plot_reliability(logits, y, T, path, n_bins=15):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, t) in zip(axes, (("Uncalibrated", 1.0), (f"Temperature scaled (T={T:.2f})", T))):
        p = softmax_np(logits, t)
        e, _, table = ece_mce(p, y, n_bins)
        xs = [r["confidence"] for r in table]; ys = [r["accuracy"] for r in table]; ns = [r["n"] for r in table]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        ax.bar([r["confidence"] for r in table], ys, width=1 / n_bins * 0.9, alpha=0.35, color="#4C72B0", label="accuracy per bin")
        ax.plot(xs, ys, "o-", color="#C44E52", label=f"ECE {e:.3f}")
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.text(x_, y_ + 0.02, str(n_), fontsize=7, ha="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.08); ax.set_xlabel("model confidence (max softmax)"); ax.set_ylabel("accuracy")
        ax.set_title(f"{name} -- test"); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
