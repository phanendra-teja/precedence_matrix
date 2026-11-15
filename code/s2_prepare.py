#!/usr/bin/env python3
"""
Stage 2 - COCO (RLE/polygon) -> Ultralytics YOLO-seg, with a GROUP-AWARE split and overlays.

    python s2_prepare.py --root D:\\precedence_matrix --out work\\yolo_seg --devices work\\devices.csv

split_mode:
  group      (default) pool every image and split by device_id (images of one phone never straddle splits);
             unknown device => each image is its own group. Guarantees every class keeps >= --min_train_images
             training images where the data allows it.
  roboflow   keep the splits exactly as exported (only valid if the export has train AND valid)
  all        every image in train (val = a copy of a few images, NOT held-out) - demo model only
RLE is decoded with pycocotools; every connected component becomes its own YOLO line; holes are counted;
every polygon is re-rasterised and compared with the source mask (round-trip IoU).
"""
import argparse, csv, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
from common import load_dataset, ann_to_mask, read_devices


def assign_splits(recs, groups, val_frac, test_frac, seed, min_train_images):
    """groups: key -> group id. Returns {image key: split}."""
    by_group = defaultdict(list)
    for r in recs:
        by_group[groups[r["key"]]].append(r)
    cls_of = {r["key"]: {x["cat_name"] for x in r["anns"]} for r in recs}
    total = Counter(c for r in recs for c in cls_of[r["key"]])
    train_cnt = Counter(total)
    gids = sorted(by_group); random.Random(seed).shuffle(gids)
    n = len(recs)
    want = {"test": int(round(test_frac * n)), "val": max(1, int(round(val_frac * n))) if val_frac > 0 else 0}
    got = {"test": 0, "val": 0}
    split_of = {r["key"]: "train" for r in recs}
    for target in ("test", "val"):
        for g in gids:
            if got[target] >= want[target]:
                break
            members = by_group[g]
            if any(split_of[r["key"]] != "train" for r in members):
                continue
            need = Counter(c for r in members for c in cls_of[r["key"]])
            if any(train_cnt[c] - k < min(min_train_images, total[c]) for c, k in need.items()):
                continue                                       # would starve a class in train
            for r in members:
                split_of[r["key"]] = target
            train_cnt.subtract(need); got[target] += len(members)
    return split_of


def palette(n):
    hsv = np.array([[[int(180 * i / max(n, 1)), 220, 255]] for i in range(n)], np.uint8)
    return [tuple(int(v) for v in c[0]) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("work/yolo_seg"), type=Path)
    ap.add_argument("--devices", type=Path)
    ap.add_argument("--split_mode", choices=["group", "roboflow", "all"], default="group")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--min_train_images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_component_px", type=float, default=4.0)
    a = ap.parse_args()

    recs, _ = load_dataset(a.root)
    dev = read_devices(a.devices)
    groups = {r["key"]: dev.get(r["file_name"], f"src::{r['source_key']}") for r in recs}
    n_known = sum(1 for r in recs if r["file_name"] in dev)
    if a.split_mode == "group":
        split_of = assign_splits(recs, groups, a.val_frac, a.test_frac, a.seed, a.min_train_images)
    elif a.split_mode == "roboflow":
        split_of = {r["key"]: r["split_yolo"] for r in recs}
        if not any(v == "val" for v in split_of.values()):
            raise SystemExit("export has no valid split; use --split_mode group")
    else:
        order = [r["key"] for r in recs]; random.Random(a.seed).shuffle(order)
        split_of = {k: "train" for k in order}
    val_copy = set(order[:max(1, len(order) // 8)]) if a.split_mode == "all" else set()

    names_all = sorted({x["cat_name"] for r in recs for x in r["anns"]})
    names, colors = names_all, palette(len(names_all))
    cid = {n: i for i, n in enumerate(names)}
    for sub in ("images", "labels", "overlays"):
        for s in ("train", "val", "test"):
            (a.out / sub / s).mkdir(parents=True, exist_ok=True)

    log, per_class_iou, totals, manifest = [], defaultdict(list), Counter(), []
    cls_split = defaultdict(Counter)
    for r in recs:
        W, H = r["width"], r["height"]
        img = cv2.imread(str(r["path"]))
        over = img.copy() if img is not None else None
        lines = []
        for x in r["anns"]:
            totals["annotations"] += 1
            m = ann_to_mask(x["seg"], H, W)
            if m is None:
                totals["skipped_undecodable"] += 1
                log.append({"image": r["key"], "ann_id": x["ann_id"], "status": "SKIPPED_undecodable"}); continue
            cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hier = hier[0] if hier is not None else []
            outers = [c for c, h in zip(cnts, hier) if h[3] == -1]
            holes = sum(1 for h in hier if h[3] != -1)
            good = [c for c in outers if len(c) >= 3 and cv2.contourArea(c) >= a.min_component_px]
            if not good:
                totals["skipped_no_valid_component"] += 1
                log.append({"image": r["key"], "ann_id": x["ann_id"], "status": "SKIPPED_no_component"}); continue
            recon = np.zeros_like(m)
            cv2.fillPoly(recon, [c.reshape(-1, 2) for c in good], 1)
            uni = int((recon | m).sum()); iou = int((recon & m).sum()) / uni if uni else 0.0
            per_class_iou[x["cat_name"]].append(iou)
            k = cid[x["cat_name"]]
            for c in good:
                pts = c.reshape(-1, 2).astype(np.float64)
                pts[:, 0] = np.clip(pts[:, 0] / W, 0, 1); pts[:, 1] = np.clip(pts[:, 1] / H, 0, 1)
                lines.append(f"{k} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
                if over is not None:
                    ov2 = over.copy(); cv2.fillPoly(ov2, [c.reshape(-1, 2)], colors[k])
                    over = cv2.addWeighted(ov2, 0.45, over, 0.55, 0)
                    cv2.polylines(over, [c.reshape(-1, 2)], True, colors[k], 2)
            totals["components_written"] += len(good); totals["holes_ignored"] += holes
            totals["components_dropped_tiny"] += len(outers) - len(good)
            log.append({"image": r["key"], "ann_id": x["ann_id"], "class": x["cat_name"], "roundtrip_iou": round(iou, 4)})
            cls_split[split_of[r["key"]]][x["cat_name"]] += 1
        targets = [split_of[r["key"]]] + (["val"] if r["key"] in val_copy else [])
        for s in targets:
            dst = a.out / "images" / s / f"{r['key']}{Path(r['file_name']).suffix}"
            shutil.copy2(r["path"], dst)
            (a.out / "labels" / s / f"{r['key']}.txt").write_text("\n".join(lines) + "\n")
            if over is not None:
                cv2.imwrite(str(a.out / "overlays" / s / f"{r['key']}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 90])
        manifest.append([r["key"], r["file_name"], r["split"], split_of[r["key"]], groups[r["key"]],
                         "DEMO: not held out" if r["key"] in val_copy else ""])

    (a.out / "dataset.yaml").write_text("\n".join(
        ["path: " + a.out.resolve().as_posix(), "train: images/train", "val: images/val", "test: images/test", "names:"]
        + [f"  {i}: {n}" for i, n in enumerate(names)]) + "\n")
    json.dump({"names": names, "split_mode": a.split_mode, "images_per_split": dict(Counter(split_of.values())),
               "instances_per_class_per_split": {s: dict(c) for s, c in cls_split.items()},
               "images_with_known_device": n_known, "images_total": len(recs)}, open(a.out / "split_summary.json", "w"), indent=2)
    json.dump({"totals": dict(totals), "roundtrip_iou_min_by_class": {k: round(float(np.min(v)), 4) for k, v in per_class_iou.items()},
               "annotations": log}, open(a.out / "conversion_log.json", "w"), indent=2)
    with open(a.out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_key", "file_name", "export_split", "assigned_split", "group", "note"]); w.writerows(manifest)

    print(f"classes: {dict(enumerate(names))}")
    print(f"images per split: {dict(Counter(split_of.values()))}   (devices known for {n_known}/{len(recs)} images)")
    print("instances per class per split:")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}", {c: cls_split[s].get(c, 0) for c in names})
    print("totals:", dict(totals))
    print("round-trip IoU (min by class):", {k: round(float(np.min(v)), 3) for k, v in per_class_iou.items()})
    miss = [c for c in names if cls_split["train"].get(c, 0) == 0]
    if miss:
        print("WARNING: classes with NO training instances:", miss)
    if n_known < len(recs):
        print("NOTE: unknown devices => each image is its own group; the same phone may appear in several splits.")
    print(f"\nLOOK at {a.out/'overlays'} (one image per class) before training.")


if __name__ == "__main__":
    main()
