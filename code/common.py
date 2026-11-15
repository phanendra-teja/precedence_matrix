"""Shared helpers: read a Roboflow COCO-segmentation export in ANY layout
(<root>/train only, train/valid/test, or a folder that just holds a COCO json)."""
import hashlib, json, re
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

SPLIT_MAP = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
SKIP_DIRS = {"outputs", "work", "data", "checkpoints", "runs", "code", "__pycache__", ".git"}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_key(file_name: str) -> str:
    """Readable stable ID from a Roboflow filename (drops the '.rf.<hash>' suffix)."""
    name = Path(file_name).name
    stem = name.split(".rf.")[0] if ".rf." in name else Path(name).stem
    return re.sub(r"[^A-Za-z0-9_\-]", "_", stem)


def ann_to_mask(seg, h, w):
    """COCO segmentation (RLE dict or polygon list) -> uint8 mask (h, w), or None."""
    try:
        if isinstance(seg, dict):
            rle = seg
            if isinstance(rle.get("counts"), list):  # uncompressed RLE
                rle = mu.frPyObjects(rle, h, w)
            m = mu.decode(rle)
        elif isinstance(seg, list) and seg:
            m = mu.decode(mu.merge(mu.frPyObjects(seg, h, w)))
        else:
            return None
    except Exception:
        return None
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.shape != (h, w):
        return None
    return np.ascontiguousarray(m.astype(np.uint8))


def find_coco_sets(root: Path, max_depth: int = 3):
    """All COCO json files under root -> [{'split','dir','json'}]. Skips our own output folders."""
    root, found = Path(root), []
    for p in sorted(root.rglob("*.json")):
        rel = p.relative_to(root)
        if len(rel.parts) > max_depth + 1 or any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            name = p.parent.name.lower()
            found.append({"split": name if name in SPLIT_MAP else "train", "dir": p.parent, "json": p})
    return found


def _resolve_image(d: Path, file_name: str):
    for cand in (d / file_name, d / "images" / file_name, d.parent / file_name, d.parent / "images" / file_name):
        if cand.exists():
            return cand
    return d / file_name


def _norm_name(n):
    return re.sub(r"\s+", " ", str(n).strip())


def _canon_key(n):
    """'Flash Light' == 'FlashLight' == 'flash light ' ; trailing plural 's' ignored."""
    k = re.sub(r"[^a-z0-9]", "", _norm_name(n).lower())
    return k[:-1] if len(k) > 3 and k.endswith("s") and not k.endswith("ss") else k


def load_aliases(root):
    """<root>/work/class_aliases.csv  (columns: from,to) - your own merges, e.g.  Connection wire,Connector"""
    p = Path(root) / "work" / "class_aliases.csv"
    out = {}
    if p.exists():
        import csv
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if (r.get("from") or "").strip() and (r.get("to") or "").strip():
                out[_norm_name(r["from"]).lower()] = _norm_name(r["to"])
    return out


def _md5(path):
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _ann_sig(a):
    return (a["cat_name"], tuple(round(v) for v in (a.get("bbox") or [])), round(a.get("area") or 0))


def load_dataset_info(root: Path):
    """(records, categories, info). Handles SEVERAL COCO json files (e.g. _annotations.coco.json, _annotations.coco_1.json ...)
    in one or many folders:
      - class names are normalised (whitespace) and spelling variants ('screw', 'Screws') are merged into the most common spelling
      - the SAME image appearing in several json files (identical bytes) becomes ONE record, annotations merged without duplicates
      - Roboflow augmentation copies (same source name, different bytes) stay separate but share `source_key`, so splits keep them together"""
    sets = find_coco_sets(root)
    if not sets:
        raise FileNotFoundError(f"No COCO json (images/annotations/categories) found under {root}")
    order = {"train": 0, "val": 1, "test": 2}
    user_alias = load_aliases(root)
    raw = []
    for s in sorted(sets, key=lambda s: (order[SPLIT_MAP[s["split"]]], str(s["json"]))):
        data = json.load(open(s["json"], encoding="utf-8"))
        local = {c["id"]: user_alias.get(_norm_name(c["name"]).lower(), _norm_name(c["name"])) for c in data.get("categories", [])}
        by_img = {}
        for a in data.get("annotations", []):
            by_img.setdefault(a["image_id"], []).append(a)
        for im in sorted(data["images"], key=lambda x: x["id"]):
            anns = [{"ann_id": a["id"], "cat_id": a["category_id"], "cat_name": local.get(a["category_id"], f"id{a['category_id']}"),
                     "seg": a.get("segmentation"), "bbox": a.get("bbox"), "area": a.get("area")}
                    for a in sorted(by_img.get(im["id"], []), key=lambda x: x["id"])]
            raw.append({"split": s["split"], "split_yolo": SPLIT_MAP[s["split"]], "json": s["json"].name,
                        "path": _resolve_image(s["dir"], im["file_name"]), "file_name": im["file_name"],
                        "source_key": image_key(im["file_name"]), "width": im["width"], "height": im["height"], "anns": anns})
    counts = {}
    for r in raw:
        for a in r["anns"]:
            counts[a["cat_name"]] = counts.get(a["cat_name"], 0) + 1
    fam = {}
    for n in counts:
        fam.setdefault(_canon_key(n), []).append(n)
    canon = {n: max(names, key=lambda x: (counts[x], x)) for names in fam.values() for n in names}
    aliases = {n: c for n, c in canon.items() if n != c}
    aliases.update({k: v for k, v in user_alias.items()})
    for r in raw:
        for a in r["anns"]:
            a["cat_name"] = canon[a["cat_name"]]
    by_hash, records, dups = {}, [], []
    for r in raw:
        h = _md5(r["path"])
        if h and h in by_hash:
            base = by_hash[h]
            seen = {_ann_sig(a) for a in base["anns"]}
            added = 0
            for a in r["anns"]:
                if _ann_sig(a) not in seen:
                    base["anns"].append(a); seen.add(_ann_sig(a)); added += 1
            dups.append({"file": r["file_name"], "also_in_json": r["json"], "first_json": base["json"],
                         "annotations_added": added, "annotations_identical": len(r["anns"]) - added})
            continue
        if h:
            by_hash[h] = r
        records.append(r)
    used = {}
    for r in records:
        k = image_key(r["file_name"]); used[k] = used.get(k, 0) + 1
        r["key"] = k if used[k] == 1 else f"{k}__c{used[k]}"
    names = sorted({a["cat_name"] for r in records for a in r["anns"]})
    info = {"json_files": sorted({r["json"] for r in raw}), "image_records_before_merge": len(raw), "images": len(records),
            "duplicate_images_merged": dups, "class_aliases_merged": aliases,
            "class_counts": dict(sorted({n: sum(1 for r in records for a in r["anns"] if a["cat_name"] == n) for n in names}.items())),
            "augmentation_copies": sum(1 for r in records if used[image_key(r["file_name"])] > 1)}
    return records, dict(enumerate(names)), info


def load_dataset(root: Path):
    """Returns (records, categories)."""
    records, cats, _ = load_dataset_info(root)
    return records, cats


def read_devices(path):
    """devices.csv (file_name, device_id[, phone_model]) -> {file_name: device_id}; blanks are skipped."""
    import csv
    out = {}
    if path and Path(path).exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            d = (r.get("device_id") or "").strip()
            if d:
                out[r["file_name"]] = d
    return out


def short_label(instance_id: str, cls: str) -> str:
    """'Screw#12' from '<image>__instance_0012'."""
    return f"{cls}#{int(instance_id.rsplit('_', 1)[1])}"
