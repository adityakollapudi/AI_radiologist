"""Four-class, derived binary (non-demented vs any dementia) and ordinal-severity metrics, plus plots."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def macro_scores(y, pred, labels=None):
    """Macro P/R/F1 over the classes present in y (classes absent from y are reported separately)."""
    labels = sorted(set(np.asarray(y).tolist())) if labels is None else labels
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_precision": float(p), "macro_recall": float(r),
            "macro_f1": float(f)}


def _safe_auc(y_bin, score):
    return float(roc_auc_score(y_bin, score)) if 0 < y_bin.sum() < len(y_bin) else float("nan")


def _safe_ap(y_bin, score):
    return float(average_precision_score(y_bin, score)) if y_bin.sum() > 0 else float("nan")


def multiclass_metrics(y, prob, class_names):
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    K = len(class_names)
    pred = prob.argmax(1)
    labels = list(range(K))
    present = sorted(set(y.tolist()))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {}
    aucs, aps, supports = [], [], []
    for k, c in enumerate(class_names):
        yb = (y == k).astype(int)
        auc, ap = _safe_auc(yb, prob[:, k]), _safe_ap(yb, prob[:, k])
        per_class[c] = {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
                        "roc_auc_ovr": auc, "pr_auc": ap}
        aucs.append(auc); aps.append(ap); supports.append(int(s[k]))
    aucs, aps, supports = np.array(aucs), np.array(aps), np.array(supports)
    ok = np.isfinite(aucs)
    top3 = np.mean([y[i] in np.argsort(-prob[i])[:3] for i in range(len(y))]) if len(y) else np.nan
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=present, average="macro", zero_division=0)
    out = {
        "n": int(len(y)),
        "classes_present_in_y": [class_names[k] for k in present],
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y, pred, labels=present, average="weighted", zero_division=0)),
        "top3_accuracy": float(top3),
        "macro_roc_auc_ovr": float(aucs[ok].mean()) if ok.any() else float("nan"),
        "weighted_roc_auc_ovr": float(np.average(aucs[ok], weights=supports[ok])) if ok.any() and supports[ok].sum() else float("nan"),
        "macro_pr_auc": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
        "log_loss": float(log_loss(y, np.clip(prob, 1e-12, 1), labels=labels)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
    cm = np.array(out["confusion_matrix"], dtype=float)
    out["normalized_confusion_matrix"] = (cm / np.clip(cm.sum(1, keepdims=True), 1, None)).round(4).tolist()
    return out


def class_specific_analysis(m, class_names):
    present = [c for c in class_names if m["per_class"][c]["support"] > 0]
    f1 = {c: m["per_class"][c]["f1"] for c in present}
    cm = np.array(m["confusion_matrix"])
    pairs = []
    for i in range(len(class_names)):
        for j in range(i + 1, len(class_names)):
            pairs.append({"pair": f"{class_names[i]} <-> {class_names[j]}", "a_to_b": int(cm[i, j]), "b_to_a": int(cm[j, i]),
                          "total": int(cm[i, j] + cm[j, i]), "severity_distance": j - i})
    pairs = sorted(pairs, key=lambda d: -d["total"])
    focus = [f"{class_names[i]} <-> {class_names[i + 1]}" for i in range(len(class_names) - 1)]
    return {"easiest_class_by_f1": max(f1, key=f1.get) if f1 else None,
            "hardest_class_by_f1": min(f1, key=f1.get) if f1 else None,
            "f1_by_class": f1, "most_confused_pairs": pairs[:5],
            "adjacent_pairs": [p for p in pairs if p["pair"] in focus]}


def binary_dementia_metrics(y, prob, class_names, threshold=0.5):
    """NON DEMENTED vs ANY DEMENTIA, derived from the four-class probabilities (P(any) = 1 - P(Non Demented))."""
    nd = class_names.index("Non Demented")
    yb = (np.asarray(y) != nd).astype(int)
    score = 1.0 - np.asarray(prob)[:, nd]
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    out = {"source": "derived from FOUR-CLASS model probabilities", "threshold": threshold, "n": int(len(yb)),
           "accuracy": div(tp + tn, len(yb)), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
           "precision": div(tp, tp + fp), "npv": div(tn, tn + fn),
           "f1": div(2 * tp, 2 * tp + fp + fn), "roc_auc": _safe_auc(yb, score), "pr_auc": _safe_ap(yb, score),
           "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "labels": ["NON DEMENTED", "ANY DEMENTIA"]}
    return out, yb, score


def ordinal_metrics(y, prob, severity):
    """severity: array mapping class index -> severity index (0..3). Analytical only, not clinical staging."""
    y = np.asarray(y); prob = np.asarray(prob)
    sev = np.asarray(severity)
    ys, ps = sev[y], sev[prob.argmax(1)]
    expected = prob @ sev
    diff = np.abs(ys - ps)
    out = {"mean_absolute_severity_error": float(diff.mean()),
           "expected_severity_mae": float(np.abs(expected - ys).mean()),
           "quadratic_weighted_kappa": float(cohen_kappa_score(ys, ps, weights="quadratic")) if len(set(ys)) > 1 else float("nan"),
           "exact": int((diff == 0).sum()), "adjacent_errors": int((diff == 1).sum()),
           "large_errors_ge2": int((diff >= 2).sum()), "n": int(len(y))}
    out["adjacent_error_share_of_errors"] = float(out["adjacent_errors"] / max(1, (diff > 0).sum()))
    under = (ps < ys).sum(); over = (ps > ys).sum()
    out["under_estimated"] = int(under); out["over_estimated"] = int(over)
    big = pd.crosstab(ys, ps)
    out["severity_confusion"] = {int(i): {int(j): int(big.loc[i, j]) for j in big.columns} for i in big.index}
    return out


# ------------------------------------------------------------------------------------------------ plots
def plot_confusion(cm, class_names, path, normalized=False, title=None):
    import matplotlib
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    rown = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    show = rown if normalized else cm
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{rown[i, j]:.1%}\n({int(cm[i, j])})" if normalized else f"{int(cm[i, j])}\n({rown[i, j]:.1%})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if show[i, j] > (show.max() * 0.55) else "black")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title or ("Normalized confusion matrix (row %)" if normalized else "Confusion matrix (counts, row %)"))
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc_pr(y, prob, class_names, roc_path, pr_path, title_suffix=""):
    import matplotlib
    import matplotlib.pyplot as plt
    y = np.asarray(y)
    for kind, path in (("roc", roc_path), ("pr", pr_path)):
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        for k, c in enumerate(class_names):
            yb = (y == k).astype(int)
            if yb.sum() == 0 or (kind == "roc" and yb.sum() == len(yb)):
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(yb, prob[:, k])
                ax.plot(fpr, tpr, lw=2, label=f"{c} (AUC {roc_auc_score(yb, prob[:, k]):.3f})")
            else:
                pr, rc, _ = precision_recall_curve(yb, prob[:, k])
                ax.plot(rc, pr, lw=2, label=f"{c} (AP {average_precision_score(yb, prob[:, k]):.3f})")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title(f"One-vs-rest ROC{title_suffix}")
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Precision-recall (one-vs-rest){title_suffix}")
        ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
        plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_binary_curves(yb, score, path):
    import matplotlib
    import matplotlib.pyplot as plt
    if not (0 < yb.sum() < len(yb)):
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    fpr, tpr, _ = roc_curve(yb, score)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC {roc_auc_score(yb, score):.3f}"); ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity"); ax[0].legend(); ax[0].set_title("Non Demented vs Any Dementia -- ROC")
    pr, rc, _ = precision_recall_curve(yb, score)
    ax[1].plot(rc, pr, lw=2, label=f"AP {average_precision_score(yb, score):.3f}"); ax[1].legend()
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("PR")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_training_curves(hist, out_dir, prefix=""):
    import matplotlib
    import matplotlib.pyplot as plt
    if hist is None or len(hist) == 0:
        return
    st1 = hist[hist.stage == 1].epoch.max() if (hist.stage == 1).any() else None
    specs = [("loss", "training_loss.png", "Loss"), ("accuracy", "training_accuracy.png", "Accuracy"),
             ("macro_f1", "training_macro_f1.png", "Macro-F1")]
    for key, fn, title in specs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(hist.epoch, hist[f"train_{key}"], "o-", label="train")
        ax.plot(hist.epoch, hist[f"val_{key}"], "o-", label="validation")
        if st1:
            ax.axvline(st1 + 0.5, color="gray", ls="--", lw=0.8, label="stage 1 | stage 2")
        ax.set_xlabel("epoch"); ax.set_title(title); ax.legend()
        plt.tight_layout(); fig.savefig(out_dir / f"{prefix}{fn}", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col in [c for c in hist.columns if c.startswith("lr_")]:
        ax.plot(hist.epoch, hist[col], "o-", label=col)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_title("Learning rate"); ax.legend()
    plt.tight_layout(); fig.savefig(out_dir / f"{prefix}learning_rate.png", dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for ax, (key, _, title) in zip(axes, specs):
        ax.plot(hist.epoch, hist[f"train_{key}"], "o-", label="train"); ax.plot(hist.epoch, hist[f"val_{key}"], "o-", label="val")
        if st1:
            ax.axvline(st1 + 0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(title); ax.legend()
    plt.tight_layout(); fig.savefig(out_dir / f"{prefix}training_curves.png", dpi=120); plt.close(fig)
