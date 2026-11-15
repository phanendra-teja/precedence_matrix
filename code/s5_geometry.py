#!/usr/bin/env python3
"""
Stage 5 - every matrix you can derive from the masks ALONE (no precedence labels needed).

    python s5_geometry.py --instances work\\instances\\instances.csv --pairs work\\instances\\pair_features.csv --out work\\geometry

Per image  (rows/columns are parts, labelled Class#N):
  distance_px.csv        gap in pixels between nearest points of two parts (0 = touching/overlapping)
  adjacency.csv          T = touching/overlapping, N = near, . = separate
  containment.csv        share of the ROW part's area that lies inside the COLUMN part's mask
  notch_filled.csv       share of the ROW part's concave 'notches' (convex hull minus mask) filled by the COLUMN part
                         -> a stacking cue: a part with a notch full of another part is partly covered/cut around by it
  direction.csv          where the COLUMN part is relative to the ROW part (N = above ... image coordinates)
  area_ratio_log10.csv   log10(area row / area column)
  nodes.csv              label <-> instance id, class, device
Across images (class level):
  class_instance_counts.csv, class_cooccurrence_matrix.csv, class_contact_matrix.csv (touching pairs),
  class_near_matrix.csv, class_adjacency_counts.csv
These are geometry facts, NOT precedence.
"""
import argparse, csv, math
from collections import defaultdict, Counter
from pathlib import Path
from common import short_label


def write_matrix(path, labels, M, fmt=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + labels)
        for u in labels:
            w.writerow([u] + [(fmt(M[u][v]) if fmt and M[u][v] != "" else M[u][v]) for v in labels])


def compass(dx, dy):
    ang = (math.degrees(math.atan2(-dy, dx)) + 360) % 360          # 0=E, 90=N (y grows downward in images)
    return ["E", "NE", "N", "NW", "W", "SW", "S", "SE"][int(((ang + 22.5) % 360) // 45)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--out", default=Path("work/geometry"), type=Path)
    a = ap.parse_args()

    inst = defaultdict(list)
    for r in csv.DictReader(open(a.instances, encoding="utf-8")):
        inst[r["image_id"]].append(r)
    pr = {}
    for r in csv.DictReader(open(a.pairs, encoding="utf-8")):
        pr[(r["instance_a_id"], r["instance_b_id"])] = r

    cls_count, cooc, contact, near = Counter(), Counter(), Counter(), Counter()
    adj_long = Counter()
    for img, rows in inst.items():
        rows = sorted(rows, key=lambda r: r["instance_id"])
        lab = {r["instance_id"]: short_label(r["instance_id"], r["class_name"]) for r in rows}
        ids = [r["instance_id"] for r in rows]; L = [lab[i] for i in ids]
        ctr = {r["instance_id"]: (float(r["centroid_x"]), float(r["centroid_y"])) for r in rows}
        d = a.out / img; d.mkdir(parents=True, exist_ok=True)
        Dm, Am, Cm, Nm, Rm, Sm = ({lab[u]: {} for u in ids} for _ in range(6))
        for u in ids:
            for v in ids:
                lu, lv = lab[u], lab[v]
                if u == v:
                    Dm[lu][lv] = 0; Am[lu][lv] = "-"; Cm[lu][lv] = 1; Nm[lu][lv] = 0; Rm[lu][lv] = "-"; Sm[lu][lv] = 0; continue
                p = pr[(u, v)]
                Dm[lu][lv] = float(p["contour_distance"])
                Am[lu][lv] = {"overlapping": "T", "touching": "T", "near": "N"}.get(p["visual_relation"], ".")
                Cm[lu][lv] = float(p["inter_over_a"]); Nm[lu][lv] = float(p["notch_a_by_b"])
                Rm[lu][lv] = compass(ctr[v][0] - ctr[u][0], ctr[v][1] - ctr[u][1])
                Sm[lu][lv] = math.log10(max(float(p["area_ratio"]), 1e-9))
        write_matrix(d / "distance_px.csv", L, Dm); write_matrix(d / "adjacency.csv", L, Am)
        write_matrix(d / "containment.csv", L, Cm, lambda x: round(x, 4)); write_matrix(d / "notch_filled.csv", L, Nm, lambda x: round(x, 4))
        write_matrix(d / "direction.csv", L, Rm); write_matrix(d / "area_ratio_log10.csv", L, Sm, lambda x: round(x, 3))
        with open(d / "nodes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["label", "instance_id", "class_name", "device_id"])
            for r in rows:
                w.writerow([lab[r["instance_id"]], r["instance_id"], r["class_name"], r["device_id"]])
        present = set()
        for r in rows:
            cls_count[r["class_name"]] += 1; present.add(r["class_name"])
        for c1 in present:
            for c2 in present:
                cooc[(c1, c2)] += 1
        for i, u in enumerate(ids):
            for v in ids[i + 1:]:
                rel = pr[(u, v)]["visual_relation"]; c1, c2 = sorted((pr[(u, v)]["class_a"], pr[(u, v)]["class_b"]))
                if rel in ("touching", "overlapping"):
                    contact[(c1, c2)] += 1; contact[(c2, c1)] += (c1 != c2); adj_long[(c1, c2, rel)] += 1
                elif rel == "near":
                    near[(c1, c2)] += 1; near[(c2, c1)] += (c1 != c2); adj_long[(c1, c2, rel)] += 1

    classes = sorted(cls_count)
    def cmat(name, C):
        with open(a.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([""] + classes)
            for c1 in classes:
                w.writerow([c1] + [C[(c1, c2)] for c2 in classes])
    cmat("class_cooccurrence_matrix.csv", cooc); cmat("class_contact_matrix.csv", contact); cmat("class_near_matrix.csv", near)
    with open(a.out / "class_instance_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class_name", "instances"]); [w.writerow([c, cls_count[c]]) for c in classes]
    with open(a.out / "class_adjacency_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class_a", "class_b", "relation", "pair_count"])
        for (c1, c2, rl), n in sorted(adj_long.items(), key=lambda x: -x[1]):
            w.writerow([c1, c2, rl, n])
    print(f"geometry matrices for {len(inst)} images -> {a.out}  (+ class-level matrices)")


if __name__ == "__main__":
    main()
