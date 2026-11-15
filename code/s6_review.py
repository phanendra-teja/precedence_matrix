#!/usr/bin/env python3
"""
Step 6: make labelling easy. For every image it draws all parts with a NUMBER (#N = the instance number),
and for every pair in annotation_template.csv it saves a zoomed crop with part A (red) and part B (blue).
It also writes pair_review.csv - open it in Excel, look at the pair image, fill `label` (+ evidence), save.

    python s6_review.py --root D:\\precedence_matrix --instances outputs\\instances\\instances.csv --out outputs\\review

Then:  python s7_precedence.py --instances work\\instances\\instances.csv --verified work\\review\\pair_review.csv

label values (leave UNKNOWN if you cannot support it):
  A_BEFORE_B            A must be removed / disconnected before B can be removed
  B_BEFORE_A            the opposite
  NO_DIRECT_DEPENDENCY  reviewed: neither blocks the other directly
evidence_type: teardown_video | service_manual | physical_inspection | other
evidence_reference: e.g. "iFixit iPhone X teardown, 03:12" or "video 7 @ 04:20"
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from common import load_dataset

TEMPLATE_COLS = ["device_id", "image_id", "instance_a_id", "instance_b_id", "operation_a", "operation_b",
                 "label", "evidence_type", "evidence_reference", "annotator", "confidence", "notes"]
HELPER_COLS = ["pair_no", "priority", "review_image", "class_a", "num_a", "class_b", "num_b"]


def num(iid):
    return int(iid.rsplit("_", 1)[1])


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    x, y = org
    for dx in (-1, 0, 1):                      # outline = offset copies, so it stays aligned on any OpenCV version
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def class_colors(names):
    names = sorted(set(names))
    hsv = np.array([[[int(179 * i / max(len(names), 1)), 200, 255]] for i in range(len(names))], np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return {n: tuple(int(v) for v in bgr[i, 0]) for i, n in enumerate(names)}


def overlay_ids(img, rows, masks, colors):
    H, W = img.shape[:2]
    lay = img.copy()
    for r in rows:
        lay[masks[r["instance_id"]]] = colors[r["class_name"]]
    out = cv2.addWeighted(lay, 0.45, img, 0.55, 0)
    for r in rows:
        cnts, _ = cv2.findContours(masks[r["instance_id"]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, colors[r["class_name"]], 1)
        put_text(out, f"{num(r['instance_id'])}", (int(float(r["centroid_x"])) - 8, int(float(r["centroid_y"])) + 5), 0.55)
    panel = np.full((H, 330, 3), 35, np.uint8)
    step = max(14, min(26, (H - 20) // max(len(rows), 1)))
    for k, r in enumerate(sorted(rows, key=lambda r: num(r["instance_id"]))):
        y = 18 + k * step
        cv2.rectangle(panel, (8, y - 11), (24, y + 3), colors[r["class_name"]], -1)
        put_text(panel, f"{num(r['instance_id'])}  {r['class_name']}", (34, y + 2), 0.5)
    return np.hstack([out, panel])


def pair_crop(img, rows_by_id, masks, a, b, relation, margin=70, minside=560):
    ra, rb = rows_by_id[a], rows_by_id[b]
    H, W = img.shape[:2]
    x0 = max(0, min(int(ra["x_min"]), int(rb["x_min"])) - margin); y0 = max(0, min(int(ra["y_min"]), int(rb["y_min"])) - margin)
    x1 = min(W, max(int(ra["x_max"]), int(rb["x_max"])) + margin); y1 = min(H, max(int(ra["y_max"]), int(rb["y_max"])) + margin)
    s = min(4.0, max(1.0, minside / max(x1 - x0, y1 - y0)))
    size = (int((x1 - x0) * s), int((y1 - y0) * s))
    crop = cv2.resize(img[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)

    def m(i):
        return cv2.resize(masks[i][y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    out = crop.copy()
    for i, r in rows_by_id.items():                       # faint context for other parts
        if i not in (a, b):
            c, _ = cv2.findContours(m(i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, c, -1, (190, 190, 190), 1)
    lay = out.copy()
    lay[m(a)] = (0, 0, 255); lay[m(b)] = (255, 120, 0)   # A red, B blue (BGR)
    out = cv2.addWeighted(lay, 0.5, out, 0.5, 0)
    for i, col, tag in ((a, (0, 0, 255), "A"), (b, (255, 120, 0), "B")):
        c, _ = cv2.findContours(m(i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c, -1, col, 2)
        ys, xs = np.nonzero(m(i))
        put_text(out, tag, (int(xs.mean()) - 6, int(ys.mean()) + 6), 0.8)
    banner = np.full((34, out.shape[1], 3), 35, np.uint8)
    put_text(banner, f"A(red): {ra['class_name']} #{num(a)}   B(blue): {rb['class_name']} #{num(b)}   [{relation}]", (6, 23), 0.5)
    return np.vstack([banner, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--template", type=Path, help="default: annotation_template.csv next to instances.csv")
    ap.add_argument("--out", default=Path("outputs/review"), type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite pair_review.csv even if it already contains labels")
    ap.add_argument("--merge_classes", default="Battery", help="comma list: classes with ONE physical part per phone; several instances of it in one image are merged into one part")
    ap.add_argument("--regroup", action="store_true", help="rewrite part_groups.csv (the old one is saved as part_groups.bak.csv)")
    a = ap.parse_args()
    inst_dir = a.instances.parent
    template = a.template or inst_dir / "annotation_template.csv"
    (a.out / "overlays_ids").mkdir(parents=True, exist_ok=True); (a.out / "pairs").mkdir(exist_ok=True)

    by_img = defaultdict(list)
    for r in csv.DictReader(open(a.instances, encoding="utf-8")):
        by_img[r["image_id"]].append(r)
    colors = class_colors([r["class_name"] for rows in by_img.values() for r in rows])
    paths = {rec["key"]: rec["path"] for rec in load_dataset(a.root)[0]}

    cache = {}
    def load(key):
        if key not in cache:
            img = cv2.imread(str(paths[key]))
            z = np.load(inst_dir / "masks" / f"{key}.npz")
            cache.clear(); cache[key] = (img, {k: z[k] for k in z.files})
        return cache[key]

    for key, rows in by_img.items():
        img, masks = load(key)
        cv2.imwrite(str(a.out / "overlays_ids" / f"{key}.jpg"), overlay_ids(img, rows, masks, colors), [cv2.IMWRITE_JPEG_QUALITY, 90])

    out_rows = []
    for n, t in enumerate(csv.DictReader(open(template, encoding="utf-8")), 1):
        key, ia, ib = t["image_id"], t["instance_a_id"], t["instance_b_id"]
        img, masks = load(key)
        rows_by_id = {r["instance_id"]: r for r in by_img[key]}
        relation = (t.get("notes") or "").split(" ")[0].replace("visual_relation=", "")
        fn = f"pairs/{n:03d}.jpg"
        cv2.imwrite(str(a.out / fn), pair_crop(img, rows_by_id, masks, ia, ib, relation), [cv2.IMWRITE_JPEG_QUALITY, 90])
        row = {**{c: t.get(c, "") for c in TEMPLATE_COLS},
               "pair_no": n, "priority": 2 if "Screw" in (rows_by_id[ia]["class_name"], rows_by_id[ib]["class_name"]) else 1,
               "review_image": fn, "class_a": rows_by_id[ia]["class_name"], "num_a": num(ia),
               "class_b": rows_by_id[ib]["class_name"], "num_b": num(ib)}
        out_rows.append(row)
    target = a.out / "pair_review.csv"
    if target.exists() and not a.force:
        done = [r for r in csv.DictReader(open(target, encoding="utf-8")) if (r.get("label") or "UNKNOWN") != "UNKNOWN"]
        if done:
            target = a.out / "pair_review.new.csv"
            print(f"WARNING: {len(done)} labels already in pair_review.csv - NOT overwritten. Wrote {target.name} instead.")
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HELPER_COLS + TEMPLATE_COLS); w.writeheader(); w.writerows(out_rows)
    parts = a.out / "part_groups.csv"
    if parts.exists() and a.regroup:
        parts.replace(a.out / "part_groups.bak.csv")
    merged = 0
    if not parts.exists():                    # one row per instance; edit part_id to merge fragments of one physical part
        merge = {c.strip().lower() for c in a.merge_classes.split(",") if c.strip()}
        with open(parts, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["image_id", "instance_id", "class_name", "part_id"])
            for key, rows in by_img.items():
                cnt = {}
                for r in rows:
                    cnt[r["class_name"].lower()] = cnt.get(r["class_name"].lower(), 0) + 1
                for r in sorted(rows, key=lambda r: num(r["instance_id"])):
                    pid = r["instance_id"]
                    if r["class_name"].lower() in merge and cnt[r["class_name"].lower()] > 1:
                        pid = f"{key}__{r['class_name'].replace(' ', '_')}_merged"; merged += 1
                    w.writerow([key, r["instance_id"], r["class_name"], pid])
    print(f"{len(by_img)} annotated overlays -> {a.out/'overlays_ids'}; {len(out_rows)} pair images -> {a.out/'pairs'}")
    print(f"Fill labels in {target} (pair image path is in review_image; priority 1 = no screw, 2 = involves a screw).")
    print(f"part_groups.csv: {merged} instances auto-merged ({a.merge_classes}); check it, and merge other split parts by giving them the same part_id.")


if __name__ == "__main__":
    main()
