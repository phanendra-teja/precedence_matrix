#!/usr/bin/env python3
"""
Stage 9 - quantitative validation of the RULES (pair rules + class stages) against pairs YOU labelled by looking at the images.

    python s9_evaluate.py --instances work\\instances\\instances.csv --verified work\\review\\pair_review.csv ^
        --sample work\\precedence\\eval_sample.csv

For every labelled pair it asks: would the rules alone have predicted it, and was that right?
Reports coverage, precision (when a rule fires), directional recall, precision per rule / per stage rule / per class pair.
--sample restricts a SECOND report to the blind random sample (eval_sample.csv) - label those first, without looking at
pair_suggestions.csv, and quote THAT precision (it is not biased by you agreeing with the engine).
--calibrate writes class_precedence.calibrated.csv (pair rules only) with measured support/confidence.
"""
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
from common import short_label
from s7_precedence import load_rules, load_stages, predict_rules, read_csv, VALID

TO_LAB = {"AB": "A_BEFORE_B", "BA": "B_BEFORE_A"}


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (honest uncertainty for small samples)."""
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]


def degenerate_warning(labels, min_n=10, share=0.9):
    """Warn when (almost) every label is the same - a sign of clicking through rather than judging each pair."""
    from collections import Counter as C
    c = C(labels)
    if len(labels) >= min_n and c and max(c.values()) / len(labels) >= share:
        top, k = c.most_common(1)[0]
        return (f"{k} of {len(labels)} labels are '{top}'. Real judgements normally mix A-first, B-first and side-by-side "
                f"(A is just the upper part in the picture). Check each pair before trusting any number below.")
    return ""


def metrics(rows, weighted=False):
    wt = (lambda r: r.get("w") or 1.0) if weighted else (lambda r: 1.0)
    W = lambda rs: sum(wt(r) for r in rs)
    n = W(rows)
    fired = [r for r in rows if r["pred"] in ("A_BEFORE_B", "B_BEFORE_A", "CONFLICT")]
    correct = [r for r in fired if r["pred"] == r["label"]]
    directional = [r for r in rows if r["label"] != "NO_DIRECT_DEPENDENCY"]
    hit = [r for r in directional if r["pred"] == r["label"]]
    per = defaultdict(Counter)
    for r in fired:
        per[r["src"]]["fired"] += 1; per[r["src"]]["correct"] += r["pred"] == r["label"]
        per[r["src"]]["opposite_direction"] += r["label"] in ("A_BEFORE_B", "B_BEFORE_A") and r["pred"] != r["label"]
        per[r["src"]]["actually_no_dependency"] += r["label"] == "NO_DIRECT_DEPENDENCY"
    return {"labelled_pairs": len(rows), "label_distribution": dict(Counter(r["label"] for r in rows)),
            "rule_fired_on": len(fired), "coverage": round(W(fired) / n, 3) if n else None,
            "precision_when_fired": round(W(correct) / W(fired), 3) if fired else None,
            "precision_95pct_interval_unweighted": wilson(len(correct), len(fired)),
            "directional_labels": len(directional), "directional_recall": round(W(hit) / W(directional), 3) if directional else None,
            "per_rule": {k: {**dict(v), "precision": round(v["correct"] / v["fired"], 3)} for k, v in per.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--verified", required=True, type=Path)
    ap.add_argument("--rules", default=Path("work/class_precedence.csv"), type=Path)
    ap.add_argument("--stages_csv", default=Path("work/class_stage.csv"), type=Path)
    ap.add_argument("--stage_gap", type=float, default=8.0)
    ap.add_argument("--sample", type=Path, help="eval_sample.csv from s7 (blind random sample)")
    ap.add_argument("--out", default=Path("work/precedence"), type=Path)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    pairs = {(r["instance_a_id"], r["instance_b_id"]): r for r in read_csv(a.pairs or a.instances.parent / "pair_features.csv")}
    rules, stages = load_rules(a.rules), load_stages(a.stages_csv)
    cls = {r["instance_id"]: r["class_name"] for r in read_csv(a.instances)}
    sample_keys = {}
    if a.sample and a.sample.exists():
        for r in read_csv(a.sample):
            sample_keys[(r["image_id"], frozenset((r["label_a"], r["label_b"])))] = float(r.get("weight") or 1.0)
    rows = []
    for r in read_csv(a.verified):
        lab = (r.get("label") or "").strip().upper()
        if lab not in VALID:
            continue
        ia, ib = r["instance_a_id"], r["instance_b_id"]
        gap = float(pairs[(ia, ib)]["contour_distance"])
        d, src = predict_rules(rules, stages, cls[ia], cls[ib], gap, a.stage_gap)
        rows.append({"image_id": r["image_id"], "a": ia, "b": ib, "ca": cls[ia], "cb": cls[ib], "gap": gap, "label": lab,
                     "pred": TO_LAB.get(d, d or ""), "src": src,
                     "in_sample": (r["image_id"], frozenset((short_label(ia, cls[ia]), short_label(ib, cls[ib])))) in sample_keys,
                     "w": sample_keys.get((r["image_id"], frozenset((short_label(ia, cls[ia]), short_label(ib, cls[ib])))))})
    if not rows:
        raise SystemExit("No labelled pairs yet - label some pairs in pair_review.csv first (label = A_BEFORE_B / B_BEFORE_A / NO_DIRECT_DEPENDENCY).")
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "rule_evaluation_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_id", "instance_a", "instance_b", "class_a", "class_b", "gap_px", "your_label", "rule_prediction", "rule", "in_blind_sample"])
        for r in rows:
            w.writerow([r["image_id"], r["a"], r["b"], r["ca"], r["cb"], r["gap"], r["label"], r["pred"], r["src"], int(r["in_sample"])])
    warn = degenerate_warning([r["label"] for r in rows])
    report = {"WARNING_labels": warn} if warn else {}
    report["all_labelled_pairs"] = metrics(rows)
    if warn:
        print("\n" + "!" * 78 + "\nWARNING: " + warn + "\n" + "!" * 78 + "\n")
    blind = [r for r in rows if r["in_sample"]]
    if a.sample:
        if blind:
            report["blind_sample"] = {"sample_size": len(sample_keys), "labelled": len(blind),
                                      "left_without_a_label": len(sample_keys) - len(blind),
                                      "unweighted": metrics(blind), "frequency_weighted": metrics(blind, weighted=True),
                                      "note": "frequency_weighted re-weights each class pair by how common it is among ALL neighbouring pairs; "
                                              "quote that one, plus how many sample pairs you could not label"}
        else:
            report["blind_sample"] = "none of the sample pairs is labelled yet"
    # per class pair, labels oriented to the sorted class order
    cp = defaultdict(lambda: Counter())
    for r in rows:
        c1, c2 = sorted((r["ca"], r["cb"])); flip = r["ca"] != c1 and r["ca"] != r["cb"]
        lab = {"A_BEFORE_B": "B_BEFORE_A", "B_BEFORE_A": "A_BEFORE_B"}.get(r["label"], r["label"]) if flip else r["label"]
        k = (c1, c2); cp[k]["n"] += 1; cp[k][lab] += 1
        if r["pred"] in ("A_BEFORE_B", "B_BEFORE_A", "CONFLICT"):
            cp[k]["rule_fired"] += 1; cp[k]["rule_correct"] += r["pred"] == r["label"]
    with open(a.out / "rule_evaluation_by_classpair.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["first_class", "second_class", "labelled", "first_before_second", "second_before_first", "no_dependency", "rule_fired", "rule_correct"])
        for (c1, c2), c in sorted(cp.items(), key=lambda x: -x[1]["n"]):
            w.writerow([c1, c2, c["n"], c["A_BEFORE_B"], c["B_BEFORE_A"], c["NO_DIRECT_DEPENDENCY"], c["rule_fired"], c["rule_correct"]])
    json.dump(report, open(a.out / "rule_evaluation.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    for (c1, c2), c in cp.items():
        if c["rule_fired"] >= 4 and c["rule_correct"] / c["rule_fired"] < 0.6:
            print(f"SUGGESTION: rules are right for only {c['rule_correct']}/{c['rule_fired']} {c1} vs {c2} pairs "
                  f"-> change a stage in class_stage.csv or add a pair rule to class_precedence.csv (see rule_evaluation_by_classpair.csv)")
    st = report["all_labelled_pairs"]["per_rule"].get("STAGE")
    if st:
        print(f"Stage rule measured precision {st['precision']} on {st['fired']} labelled pairs -> use  --stage_conf {st['precision']}  in s7_precedence.py")
    if a.calibrate and rules:
        src = read_csv(a.rules); out = a.rules.with_name(a.rules.stem + ".calibrated.csv")
        per = report["all_labelled_pairs"]["per_rule"]
        for r in src:
            v = per.get(r["rule_id"])
            if v:
                r["support"] = v["fired"]; r["confidence"] = round((v["correct"] + 1) / (v["fired"] + 2), 3); r["status"] = "CALIBRATED"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(src[0])); w.writeheader(); w.writerows(src)
        print(f"calibrated pair rules -> {out}")


if __name__ == "__main__":
    main()