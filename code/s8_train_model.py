#!/usr/bin/env python3
"""
Stage 8 - learn precedence from YOUR verified pairs (optional, but this is what generalises beyond the rules).

    python s8_train_model.py --instances work\\instances\\instances.csv --verified work\\review\\pair_review.csv --out work\\precedence_model.joblib

Features: pair geometry (distance, overlap, notch cues, relative position, size) + the two classes + contact type.
Each labelled pair gives 2 training rows (A,B) and (B,A) with flipped label; predictions average both orders.
It reports GROUP cross-validation (groups = device_id, or the image when the device is unknown), so the numbers come
from images/phones the model has not seen. Use the result with  s7_precedence.py --model ... --threshold 0.85
Refuses to train on too few labels instead of producing numbers that look meaningful.
"""
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from pfeat import featurize, feature_names, predict_pair, LABELS

VALID = {"A_BEFORE_B": "AB", "B_BEFORE_A": "BA", "NO_DIRECT_DEPENDENCY": "NONE"}
FLIP = {"AB": "BA", "BA": "AB", "NONE": "NONE"}


def make_model(kind, seed):
    if kind == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--verified", required=True, type=Path)
    ap.add_argument("--out", default=Path("work/precedence_model.joblib"), type=Path)
    ap.add_argument("--min_pairs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kind", choices=["rf", "gb"], default="rf")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import joblib
    from sklearn.model_selection import GroupKFold

    pairs_path = a.pairs or a.instances.parent / "pair_features.csv"
    inst = list(csv.DictReader(open(a.instances, encoding="utf-8")))
    classes = sorted({r["class_name"] for r in inst})
    device = {r["image_id"]: r["device_id"] for r in inst}
    pairs = {(r["instance_a_id"], r["instance_b_id"]): r for r in csv.DictReader(open(pairs_path, encoding="utf-8"))}
    items = []
    for r in csv.DictReader(open(a.verified, encoding="utf-8")):
        lab = VALID.get((r.get("label") or "").strip().upper())
        ab, ba = (r["instance_a_id"], r["instance_b_id"]), (r["instance_b_id"], r["instance_a_id"])
        if lab and ab in pairs and ba in pairs:
            g = device.get(r["image_id"], "UNKNOWN")
            items.append({"row_ab": pairs[ab], "row_ba": pairs[ba], "y": lab, "image": r["image_id"],
                          "group": r["image_id"] if g == "UNKNOWN" else g, "a": ab[0], "b": ab[1]})
    counts = Counter(i["y"] for i in items)
    print(f"verified pairs usable for training: {len(items)}  {dict(counts)}")
    if len(items) < a.min_pairs or len(counts) < 2:
        raise SystemExit(f"Need at least {a.min_pairs} verified pairs with >= 2 label types (have {len(items)}, {dict(counts)}). "
                         f"Label more pairs from work\\precedence\\next_to_label.csv, then rerun.")
    groups = np.array([i["group"] for i in items])
    k = min(a.folds, len(set(groups)))
    if k < 2:
        raise SystemExit("Need verified pairs from at least 2 different devices/images for honest cross-validation.")

    def fit(sub):
        X, y = [], []
        for i in sub:
            X += [featurize(i["row_ab"], classes), featurize(i["row_ba"], classes)]; y += [i["y"], FLIP[i["y"]]]
        m = make_model(a.kind, a.seed); m.fit(np.array(X), np.array(y)); return m

    preds = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(items, groups=groups)):
        bundle = {"model": fit([items[j] for j in tr]), "classes": classes}
        for j in te:
            p = predict_pair(bundle, items[j]["row_ab"], items[j]["row_ba"])
            best = max(p, key=p.get)
            preds.append({"image_id": items[j]["image"], "a": items[j]["a"], "b": items[j]["b"], "true": items[j]["y"],
                          "pred": best, "confidence": round(p[best], 3), "p_AB": round(p["AB"], 3), "p_BA": round(p["BA"], 3),
                          "p_NONE": round(p["NONE"], 3), "fold": fold})
    acc = float(np.mean([p["true"] == p["pred"] for p in preds]))
    maj = counts.most_common(1)[0][0]
    base = counts[maj] / len(items)
    curve = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        sel = [p for p in preds if p["confidence"] >= t]
        curve.append({"threshold": t, "coverage": round(len(sel) / len(preds), 3),
                      "accuracy": round(float(np.mean([p["true"] == p["pred"] for p in sel])), 3) if sel else None, "n": len(sel)})
    dirs = [p for p in preds if p["pred"] in ("AB", "BA")]
    dir_prec = float(np.mean([p["true"] == p["pred"] for p in dirs])) if dirs else None
    conf = {t: {q: sum(1 for p in preds if p["true"] == t and p["pred"] == q) for q in LABELS} for t in LABELS}
    report = {"n_pairs": len(items), "label_counts": dict(counts), "folds": k, "group_cv_accuracy": round(acc, 3),
              "majority_baseline": round(base, 3), "directional_precision": None if dir_prec is None else round(dir_prec, 3),
              "n_directional_predictions": len(dirs), "coverage_vs_accuracy": curve, "confusion_true_by_pred": conf}
    final = make_model(a.kind, a.seed)
    X, y = [], []
    for i in items:
        X += [featurize(i["row_ab"], classes), featurize(i["row_ba"], classes)]; y += [i["y"], FLIP[i["y"]]]
    final.fit(np.array(X), np.array(y))
    imp = sorted(zip(feature_names(classes), getattr(final, "feature_importances_", [0] * len(feature_names(classes)))), key=lambda t: -t[1])[:10]
    report["top_features"] = [(n, round(float(v), 3)) for n, v in imp]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "classes": classes, "labels": LABELS, "report": report}, a.out)
    json.dump(report, open(a.out.with_suffix(".report.json"), "w"), indent=2)
    with open(a.out.with_suffix(".cv_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0])); w.writeheader(); w.writerows(preds)
    print(json.dumps(report, indent=2))
    print(f"\nGroup-CV accuracy {acc:.3f} vs majority-class baseline {base:.3f}. Model -> {a.out}")
    if acc <= base + 0.03:
        print("WARNING: the model is not clearly better than always guessing the most common label - label more pairs / check the labels.")


if __name__ == "__main__":
    main()
