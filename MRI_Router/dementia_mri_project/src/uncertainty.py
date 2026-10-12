"""Model uncertainty: maximum softmax probability, predictive entropy and optional Monte-Carlo dropout.

Wording: "model confidence" / "model uncertainty" -- never clinical certainty."""
import numpy as np
import torch


def entropy(prob):
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def normalized_entropy(prob):
    return entropy(prob) / np.log(np.asarray(prob).shape[-1])


def uncertainty_status(conf, nent, cfg):
    u = cfg["uncertainty"]
    if conf >= u["low_confidence"] and nent <= u["low_entropy"]:
        return "LOW uncertainty"
    if conf < u["high_confidence"] or nent >= u["high_entropy"]:
        return "HIGH uncertainty"
    return "MODERATE uncertainty"


def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_probs(model, x, passes=20, T=1.0):
    """x: batch tensor on the model device. Returns (mean_prob, per-pass probs). Only Dropout layers are stochastic."""
    enable_mc_dropout(model)
    ps = torch.stack([torch.softmax(model(x).float() / T, 1) for _ in range(passes)])
    model.eval()
    return ps.mean(0), ps


def mc_dropout_summary(ps):
    """ps: (passes, N, K) numpy. Total entropy, expected entropy and mutual information (epistemic part)."""
    mean = ps.mean(0)
    total = entropy(mean)
    expected = entropy(ps).mean(0)
    return {"mean_prob": mean, "total_entropy": total, "expected_entropy": expected,
            "mutual_information": total - expected, "prob_std": ps.std(0)}


def uncertainty_table(prob, y, cfg):
    conf = prob.max(1)
    ne = normalized_entropy(prob)
    status = np.array([uncertainty_status(c, e, cfg) for c, e in zip(conf, ne)])
    correct = prob.argmax(1) == np.asarray(y)
    rows = {}
    for s in ("LOW uncertainty", "MODERATE uncertainty", "HIGH uncertainty"):
        m = status == s
        rows[s] = {"n": int(m.sum()), "share": float(m.mean()), "accuracy": float(correct[m].mean()) if m.any() else None}
    return {"by_status": rows, "mean_confidence": float(conf.mean()), "mean_normalized_entropy": float(ne.mean()),
            "mean_conf_correct": float(conf[correct].mean()) if correct.any() else None,
            "mean_conf_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
            "thresholds": cfg["uncertainty"]}, conf, ne, status


def risk_coverage(prob, y):
    conf = prob.max(1); correct = prob.argmax(1) == np.asarray(y)
    order = np.argsort(-conf)
    cov = np.arange(1, len(conf) + 1) / len(conf)
    acc = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    return cov, acc


def plot_uncertainty(prob, y, conf, nent, path):
    import matplotlib
    import matplotlib.pyplot as plt
    correct = prob.argmax(1) == np.asarray(y)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    ax[0].hist(conf[correct], bins=30, alpha=0.6, label="correct"); ax[0].hist(conf[~correct], bins=30, alpha=0.6, label="incorrect")
    ax[0].set_xlabel("model confidence (calibrated max softmax)"); ax[0].legend(); ax[0].set_title("Confidence")
    ax[1].hist(nent[correct], bins=30, alpha=0.6, label="correct"); ax[1].hist(nent[~correct], bins=30, alpha=0.6, label="incorrect")
    ax[1].set_xlabel("normalized predictive entropy"); ax[1].legend(); ax[1].set_title("Model uncertainty")
    cov, acc = risk_coverage(prob, y)
    ax[2].plot(cov, acc); ax[2].set_xlabel("coverage (most confident first)"); ax[2].set_ylabel("accuracy")
    ax[2].set_title("Accuracy vs coverage")
    plt.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
