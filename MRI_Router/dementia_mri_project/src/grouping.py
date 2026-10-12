"""STAGE 2 -- source grouping, leakage-free grouped stratified split, leakage assertions, metadata.

Group evidence priority: patient ID > study ID > examination ID > original-image filename link >
exact duplicate > validated perceptual transformation. Visual similarity alone never merges images
from two different identified patients.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .utils import SEVERITY, load_json, print_header, save_json

SPLITS = ("train", "val", "test")
METHOD_PRIORITY = ["patient_id", "study_id", "exam_id", "original_image_link", "same_filename_stem",
                   "exact_duplicate", "perceptual_transform", "singleton"]
CONFIDENCE = {"patient_id": "high", "study_id": "high", "exam_id": "high", "original_image_link": "medium",
              "same_filename_stem": "medium", "exact_duplicate": "high", "perceptual_transform": "medium",
              "singleton": "low (no relationship evidence; independence not established)"}


class LeakageError(RuntimeError):
    pass


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)
            return True
        return False

    def components(self):
        comp = defaultdict(list)
        for i in range(len(self.p)):
            comp[self.find(i)].append(i)
        return comp


# --------------------------------------------------------------------------------------------- grouping
def build_source_groups(aud, pairs, cfg):
    """Return (usable dataframe with source_group_id etc., grouping log)."""
    valid = aud.valid.fillna(False).astype(bool) & aud["class"].notna()
    usable = aud[valid & ~aud.label_conflict.fillna(False).astype(bool)].copy()
    excluded = aud[valid & aud.label_conflict.fillna(False).astype(bool)].copy()
    excluded["exclusion_reason"] = "exact duplicate with conflicting class labels"
    usable = usable.reset_index().rename(columns={"index": "audit_idx"})
    pos = {a: i for i, a in enumerate(usable.audit_idx)}
    n = len(usable)
    uf = UnionFind(n)
    evidence = defaultdict(set)              # row -> set of evidence types touching it
    log = {}

    def union_by(col, method, mask=None):
        d = usable if mask is None else usable[mask]
        cnt = 0
        for _, idx in d.dropna(subset=[col]).groupby(col).groups.items():
            idx = list(idx)
            if len(idx) > 1:
                for i in idx[1:]:
                    cnt += uf.union(idx[0], i)
            for i in idx:
                evidence[i].add(method)
        log[f"unions_{method}"] = cnt

    has_pid = usable.patient_id.notna()
    union_by("patient_id", "patient_id")
    union_by("study_id", "study_id", ~has_pid)
    union_by("exam_id", "exam_id", ~has_pid & usable.study_id.isna())

    # original <-> augmented filename families (same class, specific base key)
    fam = usable.dropna(subset=["base_key"]).groupby(["class", "base_key"])
    n_fam, cnt = 0, 0
    for _, idx in fam.groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        n_fam += 1
        marked = usable.loc[idx, "is_augmented"].any() or usable.loc[idx, "has_original_marker"].any()
        method = "original_image_link" if marked else "same_filename_stem"
        for i in idx[1:]:
            cnt += uf.union(idx[0], i)
        for i in idx:
            evidence[i].add(method)
    log["filename_families"] = n_fam
    log["unions_filename_family"] = cnt

    union_by("exact_group", "exact_duplicate")

    # validated perceptual transformations, with guards
    cand = pairs[pairs.merge_candidate & (pairs.decision != "exact_duplicate")].copy() if len(pairs) else pairs
    blocked = cand[cand.patients_differ] if len(cand) else cand
    cand = cand[~cand.patients_differ] if len(cand) else cand
    cand = cand[cand.a_idx.isin(pos) & cand.b_idx.isin(pos)] if len(cand) else cand
    log["perceptual_pairs_blocked_cross_patient"] = int(len(blocked))
    base_sizes = Counter(uf.find(i) for i in range(n))
    base_max = max(base_sizes.values()) if base_sizes else 0
    # a perceptual-merge component may not exceed max_component images nor 5% of the dataset (unless an identifier
    # group is already that large): long similarity chains across many images are not evidence of a shared source.
    # Edges that would build such a chain are dropped (and reported); all other validated edges are applied.
    cap = max(min(int(cfg["phash"]["max_component"]), int(0.05 * n)), base_max)
    pairs = pairs.copy()
    pairs["merged"] = False
    edges = cand
    trial = UnionFind(n); trial.p = list(uf.p)
    for a_, b_ in zip(edges.a_idx, edges.b_idx):
        trial.union(pos[a_], pos[b_])
    tsize = Counter(trial.find(i) for i in range(n))
    big = {r for r, k in tsize.items() if k > cap and k > base_sizes.get(r, 0)}
    keep = [(a_, b_, ix) for a_, b_, ix in zip(edges.a_idx, edges.b_idx, edges.index) if trial.find(pos[a_]) not in big]
    for a_, b_, ix in keep:
        uf.union(pos[a_], pos[b_])
        evidence[pos[a_]].add("perceptual_transform"); evidence[pos[b_]].add("perceptual_transform")
    pairs.loc[[ix for _, _, ix in keep], "merged"] = True
    log["perceptual_component_cap"] = cap
    log["perceptual_unions_applied"] = len(keep)
    log["perceptual_edges_dropped_chaining"] = int(len(edges) - len(keep))
    if len(edges) - len(keep):
        print(f"  {len(edges) - len(keep)} perceptual edges dropped: they chain into components larger than {cap} images")

    comps = uf.components()
    gid = np.empty(n, dtype=object)
    meth = np.empty(n, dtype=object)
    for k, (root, members) in enumerate(sorted(comps.items(), key=lambda kv: min(kv[1]))):
        ev = set().union(*[evidence[i] for i in members])
        if len(members) == 1 and not (ev & {"patient_id", "study_id", "exam_id"}):
            m = "singleton"
        else:
            m = next(x for x in METHOD_PRIORITY if x in ev or x == "singleton")
        for i in members:
            gid[i] = f"SG{k:06d}"
            meth[i] = m
    usable["source_group_id"] = gid
    usable["grouping_method"] = meth
    usable["grouping_confidence"] = usable.grouping_method.map(CONFIDENCE)
    usable["group_size"] = usable.groupby("source_group_id").relpath.transform("size")

    # exact-duplicate redundant copies: keep one image per identical-bytes group
    usable["excluded"] = False
    usable["exclusion_reason"] = None
    if cfg.get("dedup_exact", True):
        dup = usable.dropna(subset=["exact_group"]).sort_values("relpath")
        drop = dup[dup.duplicated("exact_group", keep="first")].index
        usable.loc[drop, "excluded"] = True
        usable.loc[drop, "exclusion_reason"] = "redundant exact-duplicate copy (one copy kept)"
    mixed = usable[~usable.excluded].groupby("source_group_id")["class"].nunique()
    log["groups_total"] = int(usable.source_group_id.nunique())
    log["groups_with_mixed_labels"] = int((mixed > 1).sum())
    log["group_methods"] = usable.drop_duplicates("source_group_id").grouping_method.value_counts().to_dict()
    log["images_by_group_method"] = usable.grouping_method.value_counts().to_dict()
    log["largest_group"] = int(usable.group_size.max()) if n else 0
    log["excluded_label_conflict_images"] = int(len(excluded))
    log["excluded_redundant_copies"] = int(usable.excluded.sum())
    return usable, excluded, log, pairs


# ------------------------------------------------------------------------------------------------ split
def _allocate(groups, totals, fr, rng, small_policy):
    target = {s: {c: fr[s] * totals[c] for c in totals} for s in SPLITS}
    cur = {s: defaultdict(int) for s in SPLITS}
    assign = {}
    limitations = []
    for c, gc in groups.groupby("label"):
        ids = list(gc.index)
        if len(ids) >= 3:
            small = gc[gc.n <= max(1.0, min(target["val"][c], target["test"][c]))]
            pool = list(small.index) if len(small) >= 2 else ids
            pick = rng.choice(pool, size=2, replace=False)
            for s, g in zip(("val", "test"), pick):
                assign[g] = s; cur[s][c] += int(groups.at[g, "n"])
        else:
            order = gc.sort_values("n", ascending=False).index.tolist()
            plan = ["train", "test"] if small_policy == "train_test" else ["train", "val"]
            for g, s in zip(order, plan):
                assign[g] = s; cur[s][c] += int(groups.at[g, "n"])
            missing = [s for s in SPLITS if s not in plan[:len(order)]]
            limitations.append(f"class '{c}' has only {len(ids)} independent source group(s); split(s) {missing} "
                               "contain no images of this class")
    rest = [g for g in groups.index if g not in assign]
    rng.shuffle(rest)
    rest.sort(key=lambda g: -groups.at[g, "n"])
    for g in rest:
        c, n = groups.at[g, "label"], int(groups.at[g, "n"])
        need = [(target[s][c] - cur[s][c]) / max(target[s][c], 1e-9) for s in SPLITS]
        s = SPLITS[int(np.argmax(need))]
        assign[g] = s; cur[s][c] += n
    score = sum(abs(cur[s][c] - target[s][c]) for s in SPLITS for c in totals) / max(1, sum(totals.values()))
    return assign, score, limitations


def grouped_stratified_split(df, cfg):
    fr = {s: float(cfg["split"][s]) for s in SPLITS}
    assert abs(sum(fr.values()) - 1) < 1e-6, fr
    g = df.groupby("source_group_id").agg(n=("relpath", "size"),
                                          label=("class", lambda x: x.value_counts().index[0]))
    totals = df["class"].value_counts().to_dict()
    best = None
    for t in range(int(cfg["split"]["n_trials"])):
        rng = np.random.RandomState(cfg["seed"] + t)
        assign, score, lim = _allocate(g, totals, fr, rng, cfg.get("small_class_policy", "train_test"))
        if best is None or score < best[1]:
            best = (assign, score, lim, t)
    assign, score, lim, t = best
    out = df.copy()
    out["split"] = out.source_group_id.map(assign)
    return out, {"allocation_error": float(score), "best_trial": t, "limitations": lim,
                 "groups_per_split": {s: int(sum(1 for v in assign.values() if v == s)) for s in SPLITS}}


# --------------------------------------------------------------------------------------------- checks
def leakage_assertions(df, pairs):
    """Hard checks. Any overlap raises LeakageError and prints the offending keys."""
    checks, problems = {}, {}
    S = {s: df[df.split == s] for s in SPLITS}
    train_groups, val_groups, test_groups = (set(S[s].source_group_id) for s in SPLITS)
    try:
        assert train_groups.isdisjoint(val_groups)
        assert train_groups.isdisjoint(test_groups)
        assert val_groups.isdisjoint(test_groups)
        checks["source_groups_disjoint"] = True
    except AssertionError:
        checks["source_groups_disjoint"] = False
        problems["source_groups"] = sorted((train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups))[:50]
    fam = lambda d: (d["class"] + "|" + d["base_key"]).dropna()          # families are linked within a class
    for col in ("sha256", "md5", "patient_id", "study_id", "exam_id", "filename_family", "exact_group"):
        sets = {s: set(fam(S[s]) if col == "filename_family" else S[s][col].dropna()) for s in SPLITS}
        ov = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
        checks[f"{col}_disjoint"] = not ov
        if ov:
            problems[col] = sorted(ov)[:50]
    if len(pairs):
        sp = dict(zip(df.audit_idx, df.split))
        crosses = lambda d: d[[sp.get(a) is not None and sp.get(b) is not None and sp.get(a) != sp.get(b)
                               for a, b in zip(d.a_idx, d.b_idx)]]
        cand = pairs[pairs.merge_candidate & ~pairs.patients_differ & (pairs.decision != "exact_duplicate")]
        merged = pairs.get("merged", pd.Series(False, index=pairs.index)).astype(bool)
        cross = crosses(cand[merged.loc[cand.index]])
        checks["info_unmerged_near_duplicate_candidates_across_splits"] = int(len(crosses(cand[~merged.loc[cand.index]])))
        checks["validated_near_duplicates_within_one_split"] = len(cross) == 0
        if len(cross):
            problems["near_duplicate_pairs"] = cross[["image_a", "image_b", "hamming"]].head(50).values.tolist()
        blk = pairs[pairs.merge_candidate & pairs.patients_differ]
        checks["info_cross_patient_near_duplicates_across_splits"] = int(sum(
            sp.get(a) is not None and sp.get(b) is not None and sp.get(a) != sp.get(b) for a, b in zip(blk.a_idx, blk.b_idx)))
    hard = [k for k, v in checks.items() if v is False]
    if hard:
        print("LEAKAGE ASSERTIONS FAILED:", hard)
        for k, v in problems.items():
            print(f"  overlapping {k}: {v[:10]}")
        raise LeakageError(f"Leakage detected ({hard}). Training is blocked until this is fixed.")
    return checks


def existing_split_leakage(df):
    """How leaky is the dataset's own train/val/test folder structure (if any)?"""
    if set(df.orig_split.unique()) <= {"none"}:
        return {"existing_split_present": False}
    out = {"existing_split_present": True, "existing_split_counts": df.orig_split.value_counts().to_dict()}
    for col, name in (("source_group_id", "source_groups"), ("exact_group", "exact_duplicate_groups"),
                      ("patient_id", "patients"), ("base_key", "filename_families")):
        sub = df.dropna(subset=[col])
        cross = sub.groupby(col).orig_split.nunique()
        out[f"{name}_crossing_existing_splits"] = int((cross > 1).sum())
        out[f"{name}_total"] = int(cross.size)
    return out


# ---------------------------------------------------------------------------------- preprocessing choice
def preprocessing_decision(df, cfg):
    near_identical = float((df.max_channel_diff_p99 <= 3).mean())
    if cfg["color_mode"] == "auto":
        color = "gray3" if near_identical >= 0.99 else "rgb"
        why = (f"{near_identical:.1%} of images have R~G~B (99th-percentile channel spread <= 3 grey levels): colour "
               "carries no extra information, so grayscale replicated to 3 channels is used; RGB is compared in the ablation"
               if color == "gray3" else
               f"only {near_identical:.1%} of images have nearly identical channels -> keep RGB; grayscale compared in the ablation")
    else:
        color, why = cfg["color_mode"], "set explicitly in config"
    margins = df[["margin_left", "margin_right", "margin_top", "margin_bottom"]].min(axis=1)
    med_margin = float(margins.median())
    if cfg["brain_crop"] == "auto":
        crop = med_margin > 0.12
        cwhy = (f"median smallest margin around the brain foreground = {med_margin:.1%} of the side "
                f"-> {'conservative crop (bounding box + 8% margin)' if crop else 'no crop needed'}")
    else:
        crop, cwhy = bool(cfg["brain_crop"]), "set explicitly in config"
    return {"color_mode": color, "color_reason": why, "rgb_near_identical_fraction": near_identical,
            "brain_crop": crop, "brain_crop_reason": cwhy, "median_min_margin": med_margin,
            "mean_abs_R_minus_G": float(df.rg_absdiff.mean()), "mean_abs_G_minus_B": float(df.gb_absdiff.mean())}


# ----------------------------------------------------------------------------------- shortcut analysis
SHORTCUT_FEATURES = ["brightness", "contrast", "file_size", "jpeg_quality_est", "black_frac", "border_mean",
                     "foreground_frac", "margin_left", "margin_right", "margin_top", "margin_bottom", "width",
                     "height", "sharpness", "hist_entropy", "rg_absdiff", "gb_absdiff", "pix_p1", "pix_p99"]


def shortcut_analysis(df, classes, seed=42):
    """Can superficial image statistics alone predict the class? Fit on TRAIN, score on VALIDATION (never test)."""
    from scipy.stats import kruskal
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, balanced_accuracy_score
    feats = [f for f in SHORTCUT_FEATURES if f in df and df[f].notna().any()]
    X = df[feats].fillna(df[feats].median())
    y = df["class"].map({c: i for i, c in enumerate(classes)})
    tr, va = df.split == "train", df.split == "val"
    out = {"features": feats}
    if tr.sum() and va.sum() and y[tr].nunique() > 1:
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        p = rf.predict(X[va])
        labs = sorted(y[va].unique())
        out["val_macro_f1_from_image_statistics"] = float(f1_score(y[va], p, average="macro", labels=labs))
        out["val_balanced_accuracy_from_image_statistics"] = float(balanced_accuracy_score(y[va], p))
        out["chance_macro_f1_approx"] = 1.0 / max(1, len(labs))
        out["feature_importance"] = dict(sorted(zip(feats, rf.feature_importances_.round(4)), key=lambda kv: -kv[1]))
    kw = {}
    for f in feats:
        groups = [df.loc[df["class"] == c, f].dropna() for c in classes if (df["class"] == c).any()]
        if len(groups) > 1 and all(len(g) > 1 for g in groups) and np.ptp(np.concatenate(groups)) > 0:
            kw[f] = float(kruskal(*groups).pvalue)
    out["kruskal_wallis_p"] = kw
    out["per_class_means"] = df.groupby("class")[feats].mean().reindex(classes).round(4).to_dict(orient="index")
    s = out.get("val_macro_f1_from_image_statistics")
    if s is not None:
        out["interpretation"] = (
            "Superficial image statistics are strongly class-informative (macro-F1 far above chance). Some of these "
            "statistics also reflect anatomy (e.g. foreground fraction, dark CSF), so this is not proof of an artefact, "
            "but class-specific preprocessing differences cannot be excluded -> reported as a dataset limitation."
            if s > 0.5 else
            "Superficial image statistics are only weakly class-informative; no dominant global-statistics shortcut found.")
    return out


# ---------------------------------------------------------------------------------------- main stage
METADATA_COLS = ["filepath", "filename", "class", "class_index", "source_group_id", "split", "width", "height",
                 "channels", "file_size", "md5", "sha256", "phash", "is_duplicate", "is_augmented", "source_type"]


def run_split(cfg, paths):
    import matplotlib
    import matplotlib.pyplot as plt
    print_header("STEP 12 - CONSTRUCT SOURCE GROUPS")
    aud = pd.read_csv(paths["audit"] / "dataset_audit.csv", low_memory=False)
    pairs = pd.read_csv(paths["audit"] / "near_duplicate_report.csv")
    mapping = load_json(paths["data"] / "class_mapping.json")
    classes = [c["name"] for c in mapping["classes"]]
    for col in ("is_augmented", "has_original_marker", "valid", "label_conflict", "is_duplicate"):
        aud[col] = aud[col].astype(str).str.lower().isin(["true", "1"])
    if len(pairs):
        for col in ("merge_candidate", "cross_class", "patients_differ"):
            pairs[col] = pairs[col].astype(str).str.lower().isin(["true", "1"])
    usable, excluded, glog, pairs = build_source_groups(aud, pairs, cfg)
    pairs.to_csv(paths["audit"] / "near_duplicate_report.csv", index=False)
    for k, v in glog.items():
        print(f"  {k:<44} {v}")
    id_cov = float(usable.patient_id.notna().mean())
    patient_level = id_cov >= 0.999
    statement = ("Patient-level grouping: every image carries a parsed patient identifier, so splits are patient-disjoint."
                 if patient_level else
                 "Patient-level independence could not be established because reliable patient identifiers were not "
                 "available in the dataset." + (f" (patient identifiers were found for only {id_cov:.1%} of images)" if id_cov > 0 else ""))
    print("\n" + statement)

    model_df = usable[~usable.excluded].copy()
    print_header("STEP 13 - LEAKAGE-FREE GROUPED STRATIFIED SPLIT")
    model_df, split_info = grouped_stratified_split(model_df, cfg)
    for lim in split_info["limitations"]:
        print("  [LIMITATION]", lim)

    print_header("STEP 14 - LEAKAGE ASSERTIONS")
    checks = leakage_assertions(model_df, pairs)
    if checks.get("info_unmerged_near_duplicate_candidates_across_splits"):
        split_info["limitations"].append(
            f"{checks['info_unmerged_near_duplicate_candidates_across_splits']} validated near-duplicate pairs lie in different splits; they were not merged "
            "because they chain into implausibly large components (see near_duplicate_report.csv, column 'merged')")
        print("  [LIMITATION]", split_info["limitations"][-1])
    for k, v in checks.items():
        print(f"  {k:<52} {v}")
    print("All leakage assertions passed.")

    old = existing_split_leakage(usable)
    model_df["class_index"] = model_df["class"].map({c: i for i, c in enumerate(classes)})
    model_df["source_type"] = np.where(model_df.is_augmented, "augmented",
                                       np.where(model_df.has_original_marker, "original", "unknown"))
    fam_aug = model_df.groupby("source_group_id").is_augmented.transform("any")
    model_df.loc[(model_df.source_type == "unknown") & fam_aug & model_df.grouping_method.eq("original_image_link"), "source_type"] = "original"
    extra = ["relpath", "orig_split", "class_folder", "severity_index", "patient_id", "study_id", "exam_id", "id_pattern",
             "base_key", "aug_evidence", "exact_group", "grouping_method", "grouping_confidence", "group_size", "audit_idx",
             "brightness", "contrast", "sharpness", "black_frac", "foreground_frac", "max_channel_diff_p99",
             "rg_absdiff", "gb_absdiff", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "margin_left", "margin_right",
             "margin_top", "margin_bottom", "jpeg_quality_est", "mode"]
    meta = model_df[METADATA_COLS + [c for c in extra if c in model_df]]
    meta.to_csv(paths["data"] / "metadata.csv", index=False)
    for s in SPLITS:
        meta[meta.split == s].to_csv(paths["data"] / f"{s}.csv", index=False)
    usable[usable.excluded].assign(stage="dedup")[["relpath", "class", "exclusion_reason", "exact_group"]].to_csv(
        paths["data"] / "excluded_images.csv", index=False)

    print_header("STEP 15 - SPLIT SUMMARY")
    tab = meta.groupby(["split", "class"]).size().unstack(fill_value=0).reindex(index=list(SPLITS), columns=classes, fill_value=0)
    pct = (tab.T / tab.sum(1)).T * 100
    print(tab.assign(total=tab.sum(1)).to_string())
    print("\nShare of each split (%):\n", (tab.sum(1) / tab.values.sum() * 100).round(2).to_string())
    print("\nClass % inside each split:\n", pct.round(2).to_string())
    groups_tab = meta.groupby(["split", "class"]).source_group_id.nunique().unstack(fill_value=0).reindex(index=list(SPLITS), columns=classes, fill_value=0)
    print("\nSource groups per split/class:\n", groups_tab.to_string())

    train_counts = tab.loc["train"]
    imb = float(train_counts.max() / max(1, train_counts[train_counts > 0].min()))
    weights = None
    if imb > cfg["class_weight_imbalance_threshold"]:
        weights = (train_counts.sum() / (len(classes) * train_counts.clip(lower=1))).round(4).to_dict()
        print(f"\nTraining set imbalance {imb:.2f} > {cfg['class_weight_imbalance_threshold']} -> training-only class weights: {weights}")
    else:
        print(f"\nTraining set imbalance ratio {imb:.2f} -> no class weights")

    print_header("STEP 16 - RGB VS GRAYSCALE / BRAIN CROP / SHORTCUT ANALYSIS")
    pre = preprocessing_decision(meta, cfg)
    for k, v in pre.items():
        print(f"  {k:<30} {v}")
    sc = shortcut_analysis(meta, classes, cfg["seed"])
    print(f"  shortcut model (image statistics only) val macro-F1: {sc.get('val_macro_f1_from_image_statistics')}")
    print("  ", sc.get("interpretation", ""))
    save_json(sc, paths["results"] / "shortcut_analysis.json")

    summary = {
        "patient_level_split": patient_level, "patient_level_statement": statement,
        "patient_id_coverage": id_cov, "grouping_log": glog, "split_info": split_info, "leakage_checks": checks,
        "existing_split_leakage": old,
        "counts": tab.to_dict(), "percent_within_split": pct.round(3).to_dict(),
        "split_share": (tab.sum(1) / tab.values.sum()).round(4).to_dict(),
        "groups_per_split_class": groups_tab.to_dict(),
        "n_images_model": int(len(meta)), "n_excluded": int(usable.excluded.sum() + len(excluded)),
        "train_imbalance_ratio": imb, "class_weights": weights, "preprocessing": pre, "classes": classes,
    }
    save_json(summary, paths["data"] / "split_summary.json")

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = tab.sum(1)
    ax.bar(list(SPLITS), sizes.values, color=["#4C72B0", "#DD8452", "#55A868"])
    for i, n in enumerate(sizes.values):
        ax.text(i, n, f"{n:,} ({n / sizes.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Leakage-free split sizes"); ax.set_ylim(0, sizes.max() * 1.15)
    plt.tight_layout(); fig.savefig(paths["results"] / "split_distribution.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.25
    for k, s in enumerate(SPLITS):
        bars = ax.bar(np.arange(len(classes)) + (k - 1) * w, tab.loc[s].values, w, label=s)
        for b, n, p in zip(bars, tab.loc[s].values, pct.loc[s].values):
            ax.text(b.get_x() + b.get_width() / 2, n, f"{n}\n{p:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes); ax.legend(); ax.set_ylim(0, tab.values.max() * 1.25)
    ax.set_title("Class distribution per split (count and % within split)")
    plt.tight_layout(); fig.savefig(paths["results"] / "split_class_distribution.png", dpi=120); plt.close(fig)

    from .report import write_leakage_report
    write_leakage_report(summary, meta, pairs, paths)
    print(f"\nSaved metadata.csv, train.csv, val.csv, test.csv to {paths['data']}")
    return meta, summary
