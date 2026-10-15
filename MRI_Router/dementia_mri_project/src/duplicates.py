"""Exact (MD5 + SHA256) and perceptual (pHash with dihedral variants + pixel check) duplicate detection."""
from collections import Counter

import numpy as np
import pandas as pd

from .data_audit import DIHEDRAL_NAMES

_M1, _M2 = np.uint64(0x5555555555555555), np.uint64(0x3333333333333333)
_M4, _H01 = np.uint64(0x0F0F0F0F0F0F0F0F), np.uint64(0x0101010101010101)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    with np.errstate(over="ignore"):
        return ((x * _H01) >> np.uint64(56)).astype(np.uint8)


def _valid(aud):
    return aud.valid.fillna(False).astype(bool) & aud["class"].notna()


# ------------------------------------------------------------------------------------------------ exact
def exact_duplicates(aud):
    aud = aud.copy()
    v = aud[_valid(aud)]
    sizes = v.groupby("sha256").size()
    md5_sizes = v.groupby("md5").size()
    dup_hashes = sizes[sizes > 1].index
    aud["exact_group"] = np.where(aud.sha256.isin(dup_hashes) & _valid(aud), "EX_" + aud.sha256.str[:12], None)
    aud["is_duplicate"] = aud.exact_group.notna()
    rows = []
    for h in dup_hashes:
        g = v[v.sha256 == h]
        splits = sorted(g.orig_split.unique())
        rows.append({"group_id": "EX_" + h[:12], "sha256": h, "md5": g.md5.iloc[0], "n_images": len(g),
                     "classes": ";".join(sorted(g["class"].unique())), "label_conflict": g["class"].nunique() > 1,
                     "orig_splits": ";".join(splits), "crosses_existing_split": len([s for s in splits if s != "none"]) > 1,
                     "files": ";".join(g.relpath)})
    groups = pd.DataFrame(rows, columns=["group_id", "sha256", "md5", "n_images", "classes", "label_conflict",
                                         "orig_splits", "crosses_existing_split", "files"])
    tt = 0
    for _, r in groups.iterrows():
        s = set(r.orig_splits.split(";"))
        tt += int("train" in s and "test" in s)
    aud["label_conflict"] = aud.exact_group.isin(set(groups.loc[groups.label_conflict, "group_id"]))
    summary = {
        "valid_labelled_images": int(len(v)),
        "unique_sha256": int(sizes.size),
        "unique_md5": int(md5_sizes.size),
        "md5_sha256_consistent": bool(sizes.size == md5_sizes.size),
        "duplicate_groups": int(len(groups)),
        "images_in_duplicate_groups": int(groups.n_images.sum()) if len(groups) else 0,
        "redundant_copies": int((groups.n_images - 1).sum()) if len(groups) else 0,
        "groups_crossing_existing_splits": int(groups.crosses_existing_split.sum()) if len(groups) else 0,
        "groups_spanning_train_and_test": tt,
        "groups_with_label_conflict": int(groups.label_conflict.sum()) if len(groups) else 0,
    }
    return aud, groups, summary


# ------------------------------------------------------------------------------------------- perceptual
def _transform_thumb(t, k):
    a = np.fliplr(t) if k >= 4 else t
    return np.rot90(a, k % 4)


def _corr(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def perceptual_duplicates(aud, dihedral, thumbs, cfg, chunk=64):
    """Compare every image's pHash with all 8 dihedral pHashes of every other image.

    Decisions (conservative):
      exact_duplicate          identical bytes (handled by the exact step)
      likely_duplicate         Hamming <= merge_threshold, identity transform, pixel corr >= threshold
      transformed_duplicate    Hamming <= merge_threshold under a flip/rotation, pixel corr >= threshold
      uncertain                Hamming <= merge_threshold but the pixel check fails -> NOT merged
      possible_near_duplicate  merge_threshold < Hamming <= report_threshold -> reported, NOT merged
    Only likely/transformed duplicates are merge candidates; grouping.py applies further guards."""
    pc = cfg["phash"]
    mt, rt = int(pc["merge_threshold"]), int(pc["report_threshold"])
    vidx = np.where(_valid(aud).values)[0]
    D = dihedral[vidx]
    n = len(vidx)
    base = D[:, 0]
    out_i, out_j, out_d, out_k = [], [], [], []
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        best = np.full((e - s, n), 255, np.uint8)
        bk = np.zeros((e - s, n), np.uint8)
        for k in range(8):
            d = popcount64(base[s:e, None] ^ D[None, :, k])
            m = d < best
            best[m] = d[m]; bk[m] = k
        for r in range(e - s):
            i = s + r
            row = best[r]
            cand = np.where(row <= rt)[0]
            cand = cand[cand > i]
            if len(cand) > pc["max_pairs_per_image"]:
                cand = cand[np.argsort(row[cand], kind="stable")[: pc["max_pairs_per_image"]]]
            for j in cand:
                out_i.append(i); out_j.append(j); out_d.append(int(row[j])); out_k.append(int(bk[r, j]))
        if len(out_i) > pc["max_report_pairs"]:
            print(f"  !! more than {pc['max_report_pairs']:,} candidate pairs -- report truncated (merge pairs kept)")
            keep = [t for t in range(len(out_i)) if out_d[t] <= mt]
            out_i, out_j, out_d, out_k = ([x[t] for t in keep] for x in (out_i, out_j, out_d, out_k))
    rows = []
    sha = aud.sha256.values; cls = aud["class"].values; rel = aud.relpath.values
    spl = aud.orig_split.values; pid = aud.patient_id.values
    for i, j, d, k in zip(out_i, out_j, out_d, out_k):
        a, b = vidx[i], vidx[j]
        if sha[a] == sha[b]:
            dec, reason, corr = "exact_duplicate", "identical SHA256", 1.0
        elif d <= mt:
            corr = _corr(thumbs[a], _transform_thumb(thumbs[b], k))
            if corr >= pc["pixel_corr_threshold"]:
                dec = "likely_duplicate" if k == 0 else "transformed_duplicate"
                reason = f"pHash Hamming {d} <= {mt} ({DIHEDRAL_NAMES[k]}) and 64x64 pixel correlation {corr:.3f}"
            else:
                dec = "uncertain"
                reason = f"pHash Hamming {d} <= {mt} but pixel correlation {corr:.3f} < {pc['pixel_corr_threshold']}"
        else:
            corr = np.nan
            dec, reason = "possible_near_duplicate", f"pHash Hamming {d} in ({mt}, {rt}] -- similar, not merged"
        pa, pb = pid[a], pid[b]
        rows.append({"a_idx": int(a), "b_idx": int(b), "image_a": rel[a], "image_b": rel[b],
                     "class_a": cls[a], "class_b": cls[b], "orig_split_a": spl[a], "orig_split_b": spl[b],
                     "hamming": d, "similarity_score": round(1 - d / 64, 4), "transform": DIHEDRAL_NAMES[k],
                     "pixel_corr": corr, "decision": dec, "reason": reason,
                     "merge_candidate": dec in ("exact_duplicate", "likely_duplicate", "transformed_duplicate"),
                     "cross_class": cls[a] != cls[b],
                     "patients_differ": bool(isinstance(pa, str) and isinstance(pb, str) and pa != pb)})
    pairs = pd.DataFrame(rows, columns=["a_idx", "b_idx", "image_a", "image_b", "class_a", "class_b", "orig_split_a",
                                        "orig_split_b", "hamming", "similarity_score", "transform", "pixel_corr",
                                        "decision", "reason", "merge_candidate", "cross_class", "patients_differ"])
    ph = pd.Series([f"{int(x):016x}" for x in base])
    repeated = ph.duplicated(keep=False)
    summary = {
        "images_hashed": int(n),
        "images_with_repeated_identical_phash": int(repeated.sum()),
        "distinct_repeated_phash_values": int(ph[repeated].nunique()),
        "pairs_reported": int(len(pairs)),
        "pairs_by_decision": dict(Counter(pairs.decision)),
        "merge_candidate_pairs": int(pairs.merge_candidate.sum()),
        "merge_candidates_cross_class": int((pairs.merge_candidate & pairs.cross_class).sum()),
        "merge_candidates_crossing_existing_splits": int((pairs.merge_candidate & (pairs.orig_split_a != pairs.orig_split_b)).sum()),
        "thresholds": {"merge_hamming": mt, "report_hamming": rt, "pixel_corr": pc["pixel_corr_threshold"]},
    }
    return pairs, summary
