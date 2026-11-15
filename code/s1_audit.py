#!/usr/bin/env python3
"""
Stage 1 - audit a Roboflow COCO-segmentation export (any layout) and write devices_template.csv.

    python s1_audit.py --root D:\\precedence_matrix --out work\\audit

Reports per-split/per-class counts, tiny objects, broken polygons/RLE, missing files, duplicate annotations,
Roboflow augmentation copies and cross-split leakage. Nothing in --root is modified.
Fill device_id (same phone => same id) in devices_template.csv and save it as work\\devices.csv.
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from common import find_coco_sets, image_key, load_dataset_info

TINY_AREA_PX = 32 * 32


def poly_problems(seg, w, h):
    probs = []
    for poly in seg:
        if len(poly) < 6 or len(poly) % 2:
            probs.append("degenerate_polygon"); continue
        xs, ys = poly[0::2], poly[1::2]
        if min(xs) < -1 or min(ys) < -1 or max(xs) > w + 1 or max(ys) > h + 1:
            probs.append("out_of_bounds")
    return probs


def audit_set(name, d, jpath):
    data = json.load(open(jpath, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {i["id"]: i for i in data["images"]}
    anns = data["annotations"]
    r = {"set": name, "json": str(jpath), "n_images": len(images), "n_annotations": len(anns)}
    disk = {p.name for p in Path(d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")}
    names = {i["file_name"] for i in images.values()}
    r["missing_files"] = sorted(names - disk)
    per_class, per_img, types, problems, tiny, seen, dup = Counter(), Counter(), Counter(), Counter(), Counter(), set(), 0
    areas = defaultdict(list)
    for a in anns:
        im = images.get(a["image_id"])
        if im is None:
            problems["annotation_without_image"] += 1; continue
        c = cats.get(a["category_id"], f"id{a['category_id']}")
        per_class[c] += 1; per_img[a["image_id"]] += 1
        seg = a.get("segmentation")
        if isinstance(seg, dict):
            types["rle"] += 1
        elif isinstance(seg, list) and seg:
            types["polygon"] += 1
            for p in poly_problems(seg, im["width"], im["height"]):
                problems[p] += 1
        else:
            types["no_segmentation"] += 1
        ar = a.get("area", 0) or 0
        areas[c].append(ar)
        tiny[c] += ar < TINY_AREA_PX
        k = (a["image_id"], a["category_id"], tuple(round(v) for v in a.get("bbox", [])))
        dup += k in seen; seen.add(k)
    r.update(per_class_instances=dict(per_class), annotation_types=dict(types), problems=dict(problems),
             duplicate_annotations=dup, empty_images=sum(1 for i in images if per_img[i] == 0),
             instances_per_image=({"min": min(per_img.get(i, 0) for i in images), "max": max(per_img.get(i, 0) for i in images),
                                   "median": median(per_img.get(i, 0) for i in images)} if images else {}),
             tiny_instances={k: v for k, v in tiny.items() if v}, median_area={k: median(v) for k, v in areas.items()},
             image_sizes=dict(Counter(f"{i['width']}x{i['height']}" for i in images.values())))
    srcs = Counter(image_key(i["file_name"]) for i in images.values())
    r["unique_source_images"] = len(srcs); r["max_copies_of_one_source"] = max(srcs.values()) if srcs else 0
    return r, images, set(srcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("work/audit"), type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sets = find_coco_sets(a.root)
    if not sets:
        raise SystemExit(f"No COCO json found under {a.root}. Expected e.g. {a.root}\\train\\_annotations.coco.json")
    reports, sources, rows, used = {}, {}, [], Counter()
    for s in sets:
        used[s["split"]] += 1
        name = s["split"] if used[s["split"]] == 1 else f"{s['split']}_{used[s['split']]}"
        rep, images, src = audit_set(name, s["dir"], s["json"])
        reports[name] = rep; sources[name] = src
    recs, _, info = load_dataset_info(a.root)
    rows = [[r["file_name"], "", "", r["split"]] for r in recs]
    names = list(sources)
    leak = {f"{names[i]}&{names[k]}": sorted(sources[names[i]] & sources[names[k]])
            for i in range(len(names)) for k in range(i + 1, len(names)) if sources[names[i]] & sources[names[k]]}
    readme = [p for p in a.root.glob("README*") if p.is_file()]
    json.dump({"sets": reports, "merged": info, "cross_split_overlap": leak, "readme_files": [p.name for p in readme]},
              open(a.out / "audit_report.json", "w"), indent=2)
    with open(a.out / "devices_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file_name", "device_id", "phone_model", "export_split"]); w.writerows(rows)
    L = []
    for n, r in reports.items():
        L += [f"=== {n}: {r['n_images']} images, {r['n_annotations']} annotations ({r['annotation_types']})",
              f"classes: {r['per_class_instances']}", f"instances/image: {r['instances_per_image']}",
              f"tiny (<{TINY_AREA_PX}px area): {r['tiny_instances']}", f"problems: {r['problems']}  duplicates: {r['duplicate_annotations']}  "
              f"empty images: {r['empty_images']}  missing files: {len(r['missing_files'])}",
              f"image sizes: {r['image_sizes']}  augmentation copies (max per source): {r['max_copies_of_one_source']}", ""]
    L += [f"=== MERGED DATASET (what the pipeline actually uses)",
          f"json files: {len(info['json_files'])}  image records before merge: {info['image_records_before_merge']}  unique images: {info['images']}",
          f"same image found in several json files (merged, annotations de-duplicated): {len(info['duplicate_images_merged'])}",
          f"class spelling variants merged: {info['class_aliases_merged'] or 'none'}",
          f"instances per class: {info['class_counts']}", f"augmentation copies (same source, different pixels): {info['augmentation_copies']}", ""]
    L.append(f"cross-split overlap: {leak or 'none'}")
    (a.out / "audit_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nNEXT: open {a.out/'devices_template.csv'}, fill device_id (same phone -> same id), save as work\\devices.csv")


if __name__ == "__main__":
    main()
