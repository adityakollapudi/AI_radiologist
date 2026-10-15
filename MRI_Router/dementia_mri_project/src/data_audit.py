"""STAGE 1 -- dataset discovery and complete per-image audit (no training happens here).

Steps: locate dataset -> inspect structure -> discover classes -> open/verify every image -> per-image
statistics, file hashes, perceptual hashes and thumbnails -> identifier / augmentation-name parsing.
"""
import glob
import hashlib
import io
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import SEVERITY, load_json, print_header, save_json

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
THUMB = 64

# ------------------------------------------------------------------------------------- class discovery
CLASS_SYNONYMS = {
    "nondemented": "Non Demented", "nondementia": "Non Demented", "nondemetned": "Non Demented",
    "verymilddementia": "Very mild Dementia", "verymilddemented": "Very mild Dementia",
    "milddementia": "Mild Dementia", "milddemented": "Mild Dementia",
    "moderatedementia": "Moderate Dementia", "moderatedemented": "Moderate Dementia",
}
SPLIT_SYNONYMS = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                  "test": "test", "testing": "test"}


def _norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def match_class(component):
    return CLASS_SYNONYMS.get(_norm(component))


# ------------------------------------------------------------------------------------- identifiers
ID_PATTERNS = [
    # (name, regex, patient group, study group or None, exam group or None)
    ("OASIS", re.compile(r"(OAS[1-4])_(\d{4})_MR(\d+)", re.I),
     lambda m: f"{m.group(1).upper()}_{m.group(2)}", lambda m: f"{m.group(1).upper()}_{m.group(2)}_MR{m.group(3)}", None),
    ("ADNI", re.compile(r"(\d{3}_S_\d{4})", re.I), lambda m: m.group(1).upper(), None, None),
    ("BIDS", re.compile(r"sub-([A-Za-z0-9]+)(?:_ses-([A-Za-z0-9]+))?"),
     lambda m: f"sub-{m.group(1)}", lambda m: f"sub-{m.group(1)}_ses-{m.group(2)}" if m.group(2) else None, None),
    ("generic", re.compile(r"(?:patient|subject|pat|pid)[-_ ]?([A-Za-z]*\d+)", re.I),
     lambda m: f"P_{m.group(1).upper()}", None, None),
]
SCAN_RX = re.compile(r"mpr-?(\d+)", re.I)


def parse_identifiers(relpath):
    """Return patient/study/exam IDs only when an explicit, documented pattern matches. Never invented."""
    name = Path(relpath).name
    for src, rx, pfn, sfn, efn in ID_PATTERNS:
        for target in (name, str(relpath)):
            m = rx.search(target)
            if m:
                scan = SCAN_RX.search(name)
                exam = None
                study = sfn(m) if sfn else None
                if study and scan:
                    exam = f"{study}_mpr{scan.group(1)}"
                return {"patient_id": pfn(m), "study_id": study, "exam_id": exam, "id_pattern": src}
    return {"patient_id": None, "study_id": None, "exam_id": None, "id_pattern": None}


# ------------------------------------------------------------------------- augmentation / original names
AUG_WORDS = (r"aug(?:mented|mentation|ment)?|rot(?:ate|ated|ation)?|flip(?:ped)?|hflip|vflip|mirror(?:ed)?|"
             r"bright(?:ness|ened)?|contrast|zoom(?:ed)?|shift(?:ed)?|shear(?:ed)?|noise|noisy|blur(?:red)?|"
             r"transform(?:ed)?|crop(?:ped)?|copy|gamma|elastic|scaled?|translated?")
AUG_RX = re.compile(rf"(?i)(?:^|[_\-\s.(])(?:{AUG_WORDS})(?:[_\-\s]?-?\d+)?(?=$|[_\-\s.)])")
ORIG_RX = re.compile(r"(?i)(?:^|[_\-\s.])(?:orig(?:inal)?)(?=$|[_\-\s.\d])")
AUGMENTOR_RX = re.compile(r"(?i)^(?:.*?_)?original_(?P<orig>.+?\.(?:jpe?g|png|bmp))_"
                          r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(?:jpe?g|png|bmp)$")
ROBOFLOW_RX = re.compile(r"(?i)^(?P<orig>.+?)_(?:jpe?g|png)\.rf\.[0-9a-f]{20,40}\.(?:jpe?g|png)$")
COPY_RX = re.compile(r"(?i)(?:\s*\(\d+\)|[_\- ]copy\d*)$")


def parse_augmentation(relpath):
    """Classify a file name as original / augmented / unknown and derive a base key for linking.

    Returns dict(is_augmented, aug_evidence, has_original_marker, base_key).
    The base key is only used for linking when it is specific (letters + digits, >= 6 characters)."""
    p = Path(relpath)
    name = p.name
    stem = p.stem
    evidence = []
    base = stem
    m = AUGMENTOR_RX.match(name)
    if m:
        evidence.append("augmentor_pattern")
        base = Path(m.group("orig")).stem
    else:
        m = ROBOFLOW_RX.match(name)
        if m:
            evidence.append("roboflow_pattern")
            base = m.group("orig")
    dir_tokens = [c.lower() for c in p.parts[:-1]]
    if any(t in ("augmented", "augmentation", "aug", "augmented_images") for t in dir_tokens):
        evidence.append("augmented_folder")
    orig_marker = bool(ORIG_RX.search(stem)) or any(t in ("original", "originals", "original_images") for t in dir_tokens)
    if not evidence and AUG_RX.search(stem):
        evidence.append("aug_token:" + AUG_RX.search(stem).group(0).strip("_-. ()"))
    if COPY_RX.search(stem):
        evidence.append("copy_suffix")
    # base key: remove augmentation / original tokens and copy suffixes
    b = COPY_RX.sub("", base)
    b = AUG_RX.sub("_", b)
    b = ORIG_RX.sub("_", b)
    b = re.sub(r"[_\-\s.]+", "_", b).strip("_").lower()
    specific = len(b) >= 6 and bool(re.search(r"\d", b)) and bool(re.search(r"[a-z]", b))
    return {"is_augmented": bool(evidence) and not (orig_marker and evidence == ["copy_suffix"]),
            "aug_evidence": ";".join(evidence), "has_original_marker": orig_marker,
            "base_key": b if specific else None}


# ----------------------------------------------------------------------------------- perceptual hashes
def _dct2(a):
    from scipy.fft import dct
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def _bits_to_u64(bits):
    v = 0
    for b in bits.ravel():
        v = (v << 1) | int(b)
    return np.uint64(v)


def phash_array(a32):
    d = _dct2(a32.astype(np.float64))[:8, :8]
    return _bits_to_u64(d > np.median(d))


def perceptual_hashes(gray_img):
    """pHash of the image, its 8 dihedral variants (identity, rot90/180/270, flips) and a dHash."""
    a32 = np.asarray(gray_img.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    variants = []
    for flip in (False, True):
        base = np.fliplr(a32) if flip else a32
        for k in range(4):
            variants.append(phash_array(np.rot90(base, k)))
    d = np.asarray(gray_img.resize((9, 8), Image.LANCZOS), dtype=np.float64)
    dh = _bits_to_u64(d[:, 1:] > d[:, :-1])
    return variants, dh


DIHEDRAL_NAMES = ["identity", "rot90", "rot180", "rot270", "hflip", "hflip_rot90", "hflip_rot180", "hflip_rot270"]


# ------------------------------------------------------------------------------------- per-image audit
def _jpeg_quality(im):
    q = getattr(im, "quantization", None)
    if not q:
        return np.nan
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)
    t = np.array(q[min(q)], float)
    if t.size != 64:
        return np.nan
    scale = float(np.mean(t / std) * 100)
    return float((200 - scale) / 2) if scale <= 100 else float(5000 / scale)


def audit_one(args):
    """Open, verify and measure ONE image. Never raises: errors are returned in the record."""
    path, rel = args
    rec = {"filepath": path, "relpath": rel, "filename": os.path.basename(path), "valid": False, "error": None}
    try:
        data = open(path, "rb").read()
        rec["file_size"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            rec["format"] = im.format
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            rec["jpeg_quality_est"] = _jpeg_quality(im) if im.format == "JPEG" else np.nan
            im.load()
            rgb = im.convert("RGB")
            gray = im.convert("L")
        if rec["width"] < 8 or rec["height"] < 8:
            raise ValueError(f"degenerate size {rec['width']}x{rec['height']}")
        a = np.asarray(rgb, dtype=np.float32)
        g = np.asarray(gray, dtype=np.float32)
        rec["channels"] = Image.getmodebands(rec["mode"])
        rec["aspect_ratio"] = rec["width"] / rec["height"]
        for i, c in enumerate("RGB"):
            rec[f"mean_{c}"] = float(a[..., i].mean())
            rec[f"std_{c}"] = float(a[..., i].std())
        rec["rg_absdiff"] = float(np.abs(a[..., 0] - a[..., 1]).mean())
        rec["gb_absdiff"] = float(np.abs(a[..., 1] - a[..., 2]).mean())
        rec["max_channel_diff_p99"] = float(np.percentile(a.max(-1) - a.min(-1), 99))
        rec["brightness"] = float(g.mean())
        rec["contrast"] = float(g.std())
        rec["pix_min"], rec["pix_max"] = float(g.min()), float(g.max())
        rec["pix_p1"], rec["pix_p99"] = float(np.percentile(g, 1)), float(np.percentile(g, 99))
        rec["black_frac"] = float((g < 10).mean())
        hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256) / g.size
        rec["hist_entropy"] = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
        lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        rec["sharpness"] = float(lap.var())
        h, w = g.shape
        bw = max(1, int(round(0.05 * min(h, w))))
        border = np.concatenate([g[:bw].ravel(), g[-bw:].ravel(), g[:, :bw].ravel(), g[:, -bw:].ravel()])
        rec["border_mean"] = float(border.mean())
        fg = g > 15
        rows, cols = np.where(fg.any(1))[0], np.where(fg.any(0))[0]
        if len(rows) and len(cols):
            rec["bbox_x0"], rec["bbox_x1"] = int(cols[0]), int(cols[-1] + 1)
            rec["bbox_y0"], rec["bbox_y1"] = int(rows[0]), int(rows[-1] + 1)
            rec["margin_left"], rec["margin_right"] = cols[0] / w, (w - cols[-1] - 1) / w
            rec["margin_top"], rec["margin_bottom"] = rows[0] / h, (h - rows[-1] - 1) / h
            rec["foreground_frac"] = float(fg.mean())
        else:
            for k in ("bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "margin_left", "margin_right", "margin_top",
                      "margin_bottom", "foreground_frac"):
                rec[k] = np.nan
        dih, dh = perceptual_hashes(gray)
        rec["phash"] = f"{int(dih[0]):016x}"
        rec["dhash"] = f"{int(dh):016x}"
        rec["_dihedral"] = [int(x) for x in dih]
        rec["_thumb"] = np.asarray(gray.resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)
        rec["valid"] = True
    except Exception as e:  # recorded, reported and excluded -- never silently ignored
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


# ------------------------------------------------------------------------------------- dataset location
def _has_class_dirs(root, max_depth=4):
    root = Path(root)
    for depth in range(0, max_depth + 1):
        for d in glob.glob(str(root / ("*/" * depth))):
            if match_class(Path(d).name):
                return True
    return False


def locate_dataset(cfg):
    """STEP 1. data_root from config -> /kaggle/input -> kagglehub download."""
    cands = []
    if cfg.get("data_root"):
        cands.append(Path(cfg["data_root"]))
    if Path("/kaggle/input").exists():
        cands += sorted(Path("/kaggle/input").glob("*")) + sorted(Path("/kaggle/input").glob("*/*")) + sorted(Path("/kaggle/input").glob("*/*/*"))
    for c in cands:
        if c.is_dir() and _has_class_dirs(c):
            return c
    if cfg.get("data_root"):
        raise FileNotFoundError(f"data_root={cfg['data_root']} does not contain recognised class folders")
    print("Dataset not found locally -> kagglehub.dataset_download(...)")
    import kagglehub
    p = Path(kagglehub.dataset_download(cfg["dataset_slug"]))
    if not _has_class_dirs(p):
        raise FileNotFoundError(f"kagglehub returned {p} but no class folders were recognised")
    return p


def inspect_structure(root, max_lines=40):
    """STEP 2. Directory tree with image counts (directories only)."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        n = sum(1 for f in filenames if Path(f).suffix.lower() in IMG_EXT)
        rel = os.path.relpath(dirpath, root)
        rows.append({"dir": rel, "depth": 0 if rel == "." else rel.count(os.sep) + 1, "images": n,
                     "other_files": len(filenames) - n})
    df = pd.DataFrame(rows)
    print_header("STEP 2 - DIRECTORY STRUCTURE")
    for _, r in df.head(max_lines).iterrows():
        print(f"  {'  ' * r.depth}{r.dir if r.dir != '.' else str(root)}  [{r.images} images, {r.other_files} other]")
    if len(df) > max_lines:
        print(f"  ... {len(df) - max_lines} more directories")
    return df


def list_images(root):
    out, others = [], []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            p = os.path.join(dirpath, f)
            (out if Path(f).suffix.lower() in IMG_EXT else others).append(p)
    return sorted(out), sorted(others)


def discover_classes(files, root):
    """STEP 3. Class and existing-split assignment from path components; nothing assumed from folder order."""
    recs, unknown = [], []
    for p in files:
        rel = os.path.relpath(p, root)
        parts = Path(rel).parts[:-1]
        cls, raw, split = None, None, None
        for comp in parts:
            c = match_class(comp)
            if c:
                cls, raw = c, comp
            s = SPLIT_SYNONYMS.get(_norm(comp))
            if s:
                split = s
        if cls is None:
            unknown.append(rel)
        recs.append({"filepath": p, "relpath": rel, "class": cls, "class_folder": raw, "orig_split": split or "none"})
    return pd.DataFrame(recs), unknown


def build_class_mapping(df, cfg):
    found = sorted(df["class"].dropna().unique(), key=lambda c: cfg["class_order"].index(c) if c in cfg["class_order"] else 99)
    missing = [c for c in cfg["class_order"] if c not in found]
    extra = [c for c in found if c not in cfg["class_order"]]
    if extra:
        raise ValueError(f"Unexpected classes found: {extra}")
    classes = [c for c in cfg["class_order"] if c in found]
    mapping = {
        "verified_from_folders": True,
        "classes": [{"index": i, "name": c, "severity_index": SEVERITY[c],
                     "folder_names_found": sorted(df.loc[df["class"] == c, "class_folder"].unique().tolist())}
                    for i, c in enumerate(classes)],
        "missing_expected_classes": missing,
        "note": "Index order follows the analytical severity order; folder names are kept exactly as found.",
    }
    return mapping, classes


# ------------------------------------------------------------------------------------------ main audit
def run_audit(cfg, paths):
    from .duplicates import exact_duplicates, perceptual_duplicates
    print_header("STEP 1 - LOCATE DATASET")
    root = locate_dataset(cfg)
    print("Dataset root:", root)
    structure = inspect_structure(root)
    files, others = list_images(root)
    print(f"Image files: {len(files):,} | non-image files: {len(others):,}")
    if not files:
        raise FileNotFoundError(f"No images under {root}")

    print_header("STEP 3 - DISCOVER CLASSES")
    disc, unknown = discover_classes(files, root)
    if unknown:
        print(f"!! {len(unknown)} images are not inside a recognised class folder (excluded): {unknown[:5]}")
    mapping, classes = build_class_mapping(disc, cfg)
    save_json(mapping, paths["data"] / "class_mapping.json")
    for c in mapping["classes"]:
        print(f"  {c['index']} = {c['name']:<20} severity {c['severity_index']} | folders {c['folder_names_found']}")
    if mapping["missing_expected_classes"]:
        print("!! expected classes not found:", mapping["missing_expected_classes"])
    print("Existing split folders:", dict(Counter(disc.orig_split)))

    print_header("STEP 4/5 - AUDIT AND VERIFY EVERY IMAGE")
    args = list(zip(disc.filepath, disc.relpath))
    workers = max(1, min(int(cfg.get("audit_workers", 4)), os.cpu_count() or 1))
    recs = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(audit_one, args, chunksize=64), 1):
                recs.append(r)
                if i % 2000 == 0:
                    print(f"  audited {i:,}/{len(args):,}")
    else:
        recs = [audit_one(a) for a in args]
    thumbs = np.zeros((len(recs), THUMB, THUMB), np.uint8)
    dihedral = np.zeros((len(recs), 8), np.uint64)
    for i, r in enumerate(recs):
        if r["valid"]:
            thumbs[i] = r.pop("_thumb")
            dihedral[i] = np.array(r.pop("_dihedral"), dtype=np.uint64)
    aud = pd.DataFrame(recs)
    aud = disc.merge(aud.drop(columns=["filepath"]), on="relpath", how="left")
    ids = aud.relpath.map(parse_identifiers).apply(pd.Series)
    augs = aud.relpath.map(parse_augmentation).apply(pd.Series)
    aud = pd.concat([aud, ids, augs], axis=1)
    aud["severity_index"] = aud["class"].map(SEVERITY)

    corrupted = aud[~aud.valid.fillna(False).astype(bool)]
    corrupted[["filepath", "relpath", "class", "orig_split", "error"]].to_csv(paths["audit"] / "corrupted_images.csv", index=False)
    n_total, n_valid = len(aud), int(aud.valid.sum())
    print(f"Total images: {n_total:,} | valid: {n_valid:,} | corrupted/unreadable: {n_total - n_valid:,} "
          f"| unlabelled (no class folder): {int(aud['class'].isna().sum())}")
    for _, r in corrupted.head(10).iterrows():
        print(f"  [corrupted] {r.relpath}: {r.error}")

    np.save(paths["audit"] / "thumbs64.npy", thumbs)
    np.save(paths["audit"] / "dihedral_phash.npy", dihedral)

    print_header("STEP 7/8 - EXACT DUPLICATES (MD5 + SHA256)")
    aud, dup_groups, dup_summary = exact_duplicates(aud)
    dup_groups.to_csv(paths["audit"] / "duplicate_groups.csv", index=False)
    for k, v in dup_summary.items():
        print(f"  {k:<44} {v}")

    print_header("STEP 9 - PERCEPTUAL (NEAR) DUPLICATES")
    pairs, ph_summary = perceptual_duplicates(aud, dihedral, thumbs, cfg)
    pairs.to_csv(paths["audit"] / "near_duplicate_report.csv", index=False)
    for k, v in ph_summary.items():
        print(f"  {k:<44} {v}")

    print_header("STEP 10/11 - ORIGINAL/AUGMENTED NAMES AND IDENTIFIERS")
    id_summary = identifier_summary(aud)
    for k, v in id_summary.items():
        print(f"  {k:<44} {v}")

    aud.to_csv(paths["audit"] / "dataset_audit.csv", index=False)
    summary = dataset_summary(aud, root, structure, others, mapping, dup_summary, ph_summary, id_summary)
    save_json(summary, paths["audit"] / "dataset_summary.json")
    plot_audit(aud, paths)
    from .report import write_audit_report
    write_audit_report(summary, aud, paths)
    print(f"\nAudit written to {paths['audit']} and {paths['reports'] / 'dataset_audit.md'}")
    return aud, summary


def identifier_summary(aud):
    v = aud[aud.valid.fillna(False).astype(bool)]
    s = {
        "images_with_patient_id": int(v.patient_id.notna().sum()),
        "unique_patient_ids": int(v.patient_id.nunique()),
        "patient_id_patterns": dict(Counter(v.id_pattern.dropna())),
        "images_with_study_id": int(v.study_id.notna().sum()),
        "unique_study_ids": int(v.study_id.nunique()),
        "images_with_exam_id": int(v.exam_id.notna().sum()),
        "unique_exam_ids": int(v.exam_id.nunique()),
        "images_flagged_augmented_by_name": int(v.is_augmented.sum()),
        "images_with_original_marker": int(v.has_original_marker.sum()),
        "augmentation_evidence": dict(Counter(e.split(":")[0] for x in v.aug_evidence.fillna("") for e in x.split(";") if e)),
    }
    s["patient_ids_per_class"] = v.groupby("class").patient_id.nunique().to_dict()
    conflicts = v.dropna(subset=["patient_id"]).groupby("patient_id")["class"].nunique()
    s["patients_with_multiple_classes"] = int((conflicts > 1).sum())
    s["patient_id_coverage"] = float(v.patient_id.notna().mean()) if len(v) else 0.0
    return s


def dataset_summary(aud, root, structure, others, mapping, dup_summary, ph_summary, id_summary):
    v = aud[aud.valid.fillna(False).astype(bool)]
    num = lambda col: {"min": float(v[col].min()), "max": float(v[col].max()), "mean": float(v[col].mean()),
                       "median": float(v[col].median())} if len(v) else {}
    cls = v["class"].value_counts()
    return {
        "dataset_root": str(root),
        "directories": int(len(structure)),
        "non_image_files": len(others),
        "total_files": int(len(aud)),
        "valid_images": int(len(v)),
        "corrupted_images": int((~aud.valid.fillna(False).astype(bool)).sum()),
        "unlabelled_images": int(aud["class"].isna().sum()),
        "final_usable_images_before_dedup": int(v["class"].notna().sum()),
        "class_count": int(cls.size),
        "class_distribution": cls.to_dict(),
        "class_percent": (cls / cls.sum() * 100).round(2).to_dict(),
        "imbalance_ratio_max_min": float(cls.max() / cls.min()) if len(cls) else None,
        "existing_split_structure": aud.groupby(["orig_split", "class"]).size().unstack(fill_value=0).to_dict(),
        "width": num("width"), "height": num("height"), "aspect_ratio": num("aspect_ratio"),
        "modes": dict(Counter(v["mode"])), "formats": dict(Counter(v["format"])),
        "file_size_bytes": num("file_size"),
        "brightness": num("brightness"), "contrast": num("contrast"),
        "channel_means": {c: float(v[f"mean_{c}"].mean()) for c in "RGB"} if len(v) else {},
        "rgb_channels_nearly_identical_frac": float((v.max_channel_diff_p99 <= 3).mean()) if len(v) else None,
        "mean_abs_R_minus_G": float(v.rg_absdiff.mean()) if len(v) else None,
        "mean_abs_G_minus_B": float(v.gb_absdiff.mean()) if len(v) else None,
        "jpeg_quality_est": num("jpeg_quality_est") if v.jpeg_quality_est.notna().any() else None,
        "black_frac": num("black_frac"),
        "margins": {k: num(k) for k in ("margin_left", "margin_right", "margin_top", "margin_bottom")},
        "class_mapping": mapping,
        "exact_duplicates": dup_summary,
        "perceptual_duplicates": ph_summary,
        "identifiers": id_summary,
    }


def plot_audit(aud, paths):
    import matplotlib
    import matplotlib.pyplot as plt
    v = aud[aud.valid.fillna(False).astype(bool) & aud["class"].notna()]
    order = [c for c in SEVERITY if c in set(v["class"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    cnt = v["class"].value_counts().reindex(order)
    bars = ax.bar(order, cnt.values, color="#4C72B0")
    for b, n in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({n / cnt.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("images"); ax.set_title("Class distribution (valid images, before de-duplication)")
    ax.set_ylim(0, cnt.max() * 1.2)
    plt.tight_layout(); fig.savefig(paths["results"] / "class_distribution.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, col in zip(axes, ("brightness", "contrast", "file_size", "black_frac")):
        for c in order:
            ax.hist(v.loc[v["class"] == c, col], bins=40, alpha=0.5, label=c)
        ax.set_title(f"{col} by class")
    axes[0].legend(fontsize=8)
    plt.tight_layout(); fig.savefig(paths["results"] / "image_statistics_by_class.png", dpi=110); plt.close(fig)
