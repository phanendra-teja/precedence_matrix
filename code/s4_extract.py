#!/usr/bin/env python3
"""
Stage 4 - instance table + pairwise geometry features (+ optional ResNet50 embeddings).

  Ground-truth masks (your 40 annotated images):
      python s4_extract.py --source gt   --root D:\\precedence_matrix --devices work\\devices.csv --out work\\instances
  Model predictions on the same images:
      python s4_extract.py --source pred --root D:\\precedence_matrix --weights work\\checkpoints\\seg\\weights\\best.pt --out work\\instances_pred
  NEW unlabelled images (the deployment path):
      python s4_extract.py --images D:\\new_phones --weights ...best.pt --out work\\infer\\instances

Outputs: instances.csv, pair_features.csv, masks/<image>.npz, annotation_template.csv (a REVIEW WORKLIST, all UNKNOWN).
Instance ids are  <image>__instance_NNNN  numbered top-to-bottom by centroid, so they are stable between runs on the same masks.
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np
from scipy import ndimage as ndi
from common import load_dataset, ann_to_mask, read_devices, image_key, IMG_EXT

PAIR_COLS = ["image_id", "device_id", "instance_a_id", "instance_b_id", "class_a", "class_b",
             "dx", "dy", "centroid_distance", "bbox_iou", "mask_iou", "contour_distance",
             "area_ratio", "a_left_of_b", "a_above_b", "visual_relation",
             "inter_over_a", "inter_over_b", "notch_a_by_b", "notch_b_by_a", "solidity_a", "solidity_b"]
INST_COLS = ["image_id", "split", "device_id", "instance_id", "source", "class_name", "confidence",
             "x_min", "y_min", "x_max", "y_max", "centroid_x", "centroid_y", "cx_norm", "cy_norm",
             "mask_area", "bbox_area", "fill_ratio", "aspect_ratio", "solidity", "dist_to_border", "dist_to_center",
             "mask_path", "embedding_path"]
TEMPLATE_COLS = ["device_id", "image_id", "instance_a_id", "instance_b_id", "operation_a", "operation_b",
                 "label", "evidence_type", "evidence_reference", "annotator", "confidence", "notes"]


def geometry(m):
    ys, xs = np.nonzero(m)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cnts))
    ha = cv2.contourArea(hull)
    sol = min(1.0, len(xs) / ha) if ha > 0 else 1.0
    return xs.min(), ys.min(), xs.max(), ys.max(), xs.mean(), ys.mean(), len(xs), sol, hull


def gap_of(inst, H, W):
    """Pixels inside the convex hull of a part but NOT in the part (its 'notches')."""
    if "gap" not in inst:
        hm = np.zeros((H, W), np.uint8); cv2.fillConvexPoly(hm, inst["hull"], 1)
        inst["gap"] = hm.astype(bool) & ~inst["mask"]; inst["gap_area"] = int(inst["gap"].sum())
    return inst["gap"]


def pair_features(img_id, device, insts, W, H, touch_px, near_px):
    n = len(insts)
    inter = np.zeros((n, n)); dist = np.zeros((n, n)); notch = np.zeros((n, n))
    for j in range(n):
        dt = None
        for i in range(j):
            bi, bj = insts[i]["bbox"], insts[j]["bbox"]
            x0, y0, x1, y1 = max(bi[0], bj[0]), max(bi[1], bj[1]), min(bi[2], bj[2]), min(bi[3], bj[3])
            k, overlap_box = 0, x0 <= x1 and y0 <= y1
            if overlap_box:
                k = np.logical_and(insts[i]["mask"][y0:y1 + 1, x0:x1 + 1], insts[j]["mask"][y0:y1 + 1, x0:x1 + 1]).sum()
            inter[i, j] = inter[j, i] = k
            if k > 0:
                d = 0.0
            else:
                if dt is None:
                    dt = ndi.distance_transform_edt(~insts[j]["mask"])
                d = float(dt[insts[i]["mask"]].min())
            dist[i, j] = dist[j, i] = d
            if overlap_box and d <= near_px:               # notch cue only matters for neighbouring parts
                sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
                gi, gj = gap_of(insts[i], H, W), gap_of(insts[j], H, W)
                if insts[i]["gap_area"]:
                    notch[i, j] = np.logical_and(gi[sl], insts[j]["mask"][sl]).sum() / insts[i]["gap_area"]
                if insts[j]["gap_area"]:
                    notch[j, i] = np.logical_and(gj[sl], insts[i]["mask"][sl]).sum() / insts[j]["gap_area"]
    rows, diag = [], np.hypot(W, H)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A, B = insts[i], insts[j]
            bx0, by0 = max(A["bbox"][0], B["bbox"][0]), max(A["bbox"][1], B["bbox"][1])
            bx1, by1 = min(A["bbox"][2], B["bbox"][2]), min(A["bbox"][3], B["bbox"][3])
            bi = max(0, bx1 - bx0 + 1) * max(0, by1 - by0 + 1)
            bu = A["bbox_area"] + B["bbox_area"] - bi
            mu_ = A["area"] + B["area"] - inter[i, j]
            d = dist[i, j]
            rel = "overlapping" if inter[i, j] > 0 else "touching" if d <= touch_px else "near" if d <= near_px else "separate"
            dx, dy = (B["cx"] - A["cx"]) / W, (B["cy"] - A["cy"]) / H
            rows.append([img_id, device, A["id"], B["id"], A["cls"], B["cls"], round(dx, 5), round(dy, 5),
                         round(np.hypot(dx * W, dy * H) / diag, 5), round(bi / bu, 5) if bu else 0,
                         round(inter[i, j] / mu_, 5) if mu_ else 0, round(d, 2), round(A["area"] / B["area"], 4),
                         int(A["cx"] < B["cx"]), int(A["cy"] < B["cy"]), rel,
                         round(inter[i, j] / A["area"], 5), round(inter[i, j] / B["area"], 5),
                         round(notch[i, j], 5), round(notch[j, i], 5), round(A["solidity"], 4), round(B["solidity"], 4)])
    return rows


def make_embedder(device):
    import torch, torchvision
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity(); net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(crops):
        out = []
        for k in range(0, len(crops), 32):
            x = torch.from_numpy(np.stack(crops[k:k + 32])).to(device).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                out.append(net((x - mean) / std).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048))
    return embed


def crop_for_embedding(rgb, inst, size=224):
    x0, y0, x1, y1 = inst["bbox"]
    crop = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    fill = np.array([124, 116, 104], np.uint8)
    crop[~inst["mask"][y0:y1 + 1, x0:x1 + 1]] = fill
    h, w = crop.shape[:2]; s = max(h, w)
    sq = np.full((s, s, 3), fill, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)


def records_from_folder(folder):
    recs, used = [], set()
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMG_EXT:
            im = cv2.imread(str(p))
            if im is None:
                continue
            key = image_key(p.name)
            while key in used:
                key += "_x"
            used.add(key)
            recs.append({"split": "infer", "path": p, "file_name": p.name, "key": key,
                         "width": im.shape[1], "height": im.shape[0], "anns": []})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="COCO export (for --source gt / pred on the annotated images)")
    ap.add_argument("--images", type=Path, help="folder of NEW unlabelled images (implies --source pred)")
    ap.add_argument("--source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--out", default=Path("work/instances"), type=Path)
    ap.add_argument("--devices", type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--touch_px", type=float, default=3.0)
    ap.add_argument("--near_px", type=float, default=40.0)
    ap.add_argument("--all_pairs", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    a = ap.parse_args()
    if a.images:
        a.source = "pred"
    if not a.root and not a.images:
        ap.error("give --root (annotated export) or --images (new images)")
    if a.source == "pred" and not a.weights:
        ap.error("--source pred needs --weights")
    (a.out / "masks").mkdir(parents=True, exist_ok=True)

    dev = read_devices(a.devices)
    model = None
    if a.source == "pred":
        from ultralytics import YOLO
        model = YOLO(str(a.weights))
    embed = make_embedder("cuda" if a.device != "cpu" else "cpu") if a.embeddings else None
    recs = records_from_folder(a.images) if a.images else load_dataset(a.root)[0]

    inst_rows, pair_rows, tmpl_rows, emb = [], [], [], {}
    for r in recs:
        W, H, key = r["width"], r["height"], r["key"]
        device = dev.get(r["file_name"]) or "UNKNOWN"
        raw = []
        if a.source == "gt":
            for x in r["anns"]:
                m = ann_to_mask(x["seg"], H, W)
                if m is not None and m.any():
                    raw.append((x["cat_name"], 1.0, m.astype(bool)))
        else:
            res = model.predict(str(r["path"]), imgsz=a.imgsz, conf=a.conf, retina_masks=True, verbose=False, device=a.device)[0]
            if res.masks is not None:
                for mk, c, cf in zip(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    mk = (mk > 0.5).astype(np.uint8)
                    if mk.shape != (H, W):
                        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
                    if mk.any():
                        raw.append((res.names[int(c)], float(cf), mk.astype(bool)))
        insts = []
        for cls, cf, m in raw:
            x0, y0, x1, y1, cx, cy, area, sol, hull = geometry(m)
            insts.append({"cls": cls, "conf": cf, "mask": m, "bbox": (int(x0), int(y0), int(x1), int(y1)), "cx": float(cx),
                          "cy": float(cy), "area": int(area), "solidity": float(sol), "hull": hull,
                          "bbox_area": int((x1 - x0 + 1) * (y1 - y0 + 1))})
        insts.sort(key=lambda d: (d["cy"], d["cx"]))
        for n, d in enumerate(insts, 1):
            d["id"] = f"{key}__instance_{n:04d}"
        if not insts:
            print(f"{key}: no instances"); continue
        np.savez_compressed(a.out / "masks" / f"{key}.npz", **{d["id"]: d["mask"] for d in insts})
        if embed is not None:
            rgb = cv2.cvtColor(cv2.imread(str(r["path"])), cv2.COLOR_BGR2RGB)
            for d, v in zip(insts, embed([crop_for_embedding(rgb, d) for d in insts])):
                emb[d["id"]] = v
        for d in insts:
            x0, y0, x1, y1 = d["bbox"]; bw, bh = x1 - x0 + 1, y1 - y0 + 1
            inst_rows.append([key, r["split"], device, d["id"], a.source, d["cls"], round(d["conf"], 4), x0, y0, x1, y1,
                              round(d["cx"], 2), round(d["cy"], 2), round(d["cx"] / W, 5), round(d["cy"] / H, 5),
                              d["area"], d["bbox_area"], round(d["area"] / d["bbox_area"], 4), round(bw / bh, 4),
                              round(d["solidity"], 4), round(min(x0, y0, W - 1 - x1, H - 1 - y1), 1),
                              round(float(np.hypot(d["cx"] - W / 2, d["cy"] - H / 2)), 1),
                              f"masks/{key}.npz#{d['id']}", f"embeddings.npz#{d['id']}" if embed else ""])
        rows = pair_features(key, device, insts, W, H, a.touch_px, a.near_px)
        pair_rows += rows
        seen = set()
        for p in rows:
            u = tuple(sorted((p[2], p[3])))
            if u in seen or (p[15] == "separate" and not a.all_pairs):
                continue
            seen.add(u)
            tmpl_rows.append([device, key, u[0], u[1], "", "", "UNKNOWN", "", "", "", "",
                              f"visual_relation={p[15]} (review worklist only; NOT evidence)"])
        print(f"{key}: {len(insts)} instances, {len(rows)} ordered pairs")

    def dump(name, cols, rows):
        with open(a.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    dump("instances.csv", INST_COLS, inst_rows); dump("pair_features.csv", PAIR_COLS, pair_rows)
    dump("annotation_template.csv", TEMPLATE_COLS, tmpl_rows)
    if embed is not None:
        np.savez_compressed(a.out / "embeddings.npz", **emb)
    json.dump({"source": a.source, "images": len(recs), "instances": len(inst_rows), "ordered_pairs": len(pair_rows),
               "template_pairs": len(tmpl_rows)}, open(a.out / "extract_summary.json", "w"), indent=2)
    print(f"\nWrote {a.out}: instances.csv ({len(inst_rows)}), pair_features.csv ({len(pair_rows)}), "
          f"annotation_template.csv ({len(tmpl_rows)} adjacent pairs)")


if __name__ == "__main__":
    main()
