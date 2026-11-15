"""Pair features shared by the precedence model trainer and the precedence engine."""
import math

NUM = ["dx", "dy", "centroid_distance", "bbox_iou", "mask_iou", "contour_distance", "a_left_of_b", "a_above_b",
       "inter_over_a", "inter_over_b", "notch_a_by_b", "notch_b_by_a", "solidity_a", "solidity_b"]
RELS = ["overlapping", "touching", "near", "separate"]
LABELS = ["AB", "BA", "NONE"]          # AB: row's first part before second; BA: second before first; NONE: no direct dependency


def featurize(row, classes):
    v = [float(row[c]) for c in NUM]
    v.append(math.log10(max(float(row["area_ratio"]), 1e-9)))
    v += [1.0 if row["class_a"] == c else 0.0 for c in classes]
    v += [1.0 if row["class_b"] == c else 0.0 for c in classes]
    v += [1.0 if row["visual_relation"] == r else 0.0 for r in RELS]
    return v


def feature_names(classes):
    return NUM + ["area_ratio_log10"] + [f"A={c}" for c in classes] + [f"B={c}" for c in classes] + [f"rel={r}" for r in RELS]


def predict_pair(bundle, row_ab, row_ba):
    """Symmetric probabilities {AB, BA, NONE} for an unordered pair, from both ordered rows."""
    m, cl = bundle["model"], bundle["classes"]
    p1 = dict(zip(m.classes_, m.predict_proba([featurize(row_ab, cl)])[0]))
    p2 = dict(zip(m.classes_, m.predict_proba([featurize(row_ba, cl)])[0]))
    g = lambda p, k: float(p.get(k, 0.0))
    return {"AB": (g(p1, "AB") + g(p2, "BA")) / 2, "BA": (g(p1, "BA") + g(p2, "AB")) / 2,
            "NONE": (g(p1, "NONE") + g(p2, "NONE")) / 2}
