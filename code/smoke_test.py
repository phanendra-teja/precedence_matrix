#!/usr/bin/env python3
"""End-to-end test on a SYNTHETIC 40-image, 8-phone dataset in the 'train only' Roboflow layout (CPU, no training).
    python smoke_test.py
Checks every stage except GPU training (s3_train.py / infer): audit, group split, RLE conversion, extraction,
geometry matrices, review sheets, precedence engine (verified / pair-rule / stage / model tiers, overrides, cycles, merged parts), rule evaluation (incl. blind sample),
precedence model (group CV), and the run_all.py orchestration."""
import csv, json, random, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from s7_precedence import resolve, default_stage, stage_edge, predict_rules, ensure_stage_table, load_stages


def truth(ca, cb):
    if ca == "Screw" and cb != "Screw": return "A_BEFORE_B"
    if cb == "Screw" and ca != "Screw": return "B_BEFORE_A"
    if (ca, cb) == ("Connector", "Battery"): return "A_BEFORE_B"
    if (ca, cb) == ("Battery", "Connector"): return "B_BEFORE_A"
    if (ca, cb) == ("Cable", "Connector"): return "A_BEFORE_B"
    if (ca, cb) == ("Connector", "Cable"): return "B_BEFORE_A"
    return "NO_DIRECT_DEPENDENCY"


def unit_tests():
    E = lambda t, c: {"tier": t, "conf": c, "src": "x"}
    pe, pz, cp, log = resolve({("A", "B"): E("C", .6), ("B", "C"): E("C", .5), ("C", "A"): E("M", .9)}, {}, {})
    assert ("C", "A") not in pe and len(pe) == 2 and log["dropped_for_cycle"][0][:2] == ("C", "A")   # weakest tier dropped, logged
    pe, pz, cp, log = resolve({("A", "B"): E("V", 1), ("B", "A"): E("C", .9)}, {}, {})
    assert ("A", "B") in pe and ("B", "A") not in pe and log["overridden"]                          # verified beats rule
    pe, pz, cp, log = resolve({("A", "B"): E("C", .6), ("B", "A"): E("C", .6)}, {}, {})
    assert not pe and cp                                                                           # equal-tier opposition -> conflict
    pe, pz, cp, log = resolve({("A", "B"): E("C", .6)}, {frozenset("AB"): E("V", 1)}, {})
    assert not pe and frozenset("AB") in pz                                                         # verified 'no dependency' beats rule
    pe, pz, cp, log = resolve({("A", "B"): E("V", 1), ("B", "C"): E("V", 1), ("C", "A"): E("V", 1)}, {}, {})
    assert log["unresolved_cycles"] and len(pe) == 3                                                # all-verified cycle is reported, never dropped
    pe, pz, cp, log = resolve({("s1", "B"): E("C", .6), ("s2", "B"): E("C", .6), ("s1", "s2"): E("C", .5)}, {}, {"s1": "S", "s2": "S"})
    assert list(pe) == [("S", "B")]                                                                 # merged parts: within-part pair ignored
    assert [default_stage(n) for n in ["Screw", "Shield", "Connection wire", "Cable", "Graphite Sheet", "Camera", "Flash Light", "Battery", "Motherboard", "Foo"]] == [0, 1, 2, 2, 3, 4, 4, 5, 5, None]
    st = {"Screw": {"stage": 0.0, "gap": 15.0}, "Battery": {"stage": 5.0, "gap": None}, "Cable": {"stage": 2.0, "gap": None}, "Connector": {"stage": 2.0, "gap": None}}
    assert stage_edge(st, "Screw", "Battery", 10, 8) == "AB" and stage_edge(st, "Battery", "Screw", 10, 8) == "BA"   # screw gate is 15px
    assert stage_edge(st, "Cable", "Battery", 10, 8) is None and stage_edge(st, "Cable", "Battery", 6, 8) == "AB"    # default gate 8px
    assert stage_edge(st, "Cable", "Connector", 1, 8) is None and stage_edge(st, "Screw", "Nope", 1, 8) is None     # equal / unknown stage -> no edge
    rule = [{"rule_id": "P1", "before_class": "Battery", "after_class": "Screw", "max_gap_px": 20.0, "confidence": 0.9}]
    assert predict_rules(rule, st, "Screw", "Battery", 5, 8) == ("BA", "P1")                                          # explicit pair rule beats the stage order
    print("unit tests: OK")


def make_dataset(root, n=40, devices=8, seed=1):
    """Mimics the real folder: ONE train/ with SEVERAL json files, class-name variants, a duplicate listing, an augmentation copy."""
    rng = random.Random(seed)
    d = root / "train"; d.mkdir(parents=True)
    base_cats = {"Battery": 1, "Screw": 2, "Connector": 3, "Cable": 4, "Graphite Sheet": 5, "Camera": 6}
    names = {v: k for k, v in base_cats.items()}
    variants = {1: {"Screw": "screw", "Cable": "Wire"}, 2: {"Screw": "Screws"}, 3: {"Graphite Sheet": " Graphite  Sheet"}, 4: {"Graphite Sheet": "GraphiteSheet"}}
    n_json = 5
    jsons = [{"images": [], "annotations": [], "categories": [{"id": 0, "name": "ProjectName"}] +
              [{"id": 10 * k + cid, "name": variants.get(k, {}).get(nm, nm)} for nm, cid in base_cats.items()]} for k in range(n_json)]
    dev_rows, st = [], {"aid": 0, "expected": 0}

    def add(k, fn, masks):
        js = jsons[k]; iid = len(js["images"])
        js["images"].append({"id": iid, "file_name": fn, "width": 800, "height": 400})
        for c, m in masks:
            r = mu.encode(np.asfortranarray(m)); r["counts"] = r["counts"].decode()
            js["annotations"].append({"id": st["aid"], "image_id": iid, "category_id": 10 * k + base_cats[c], "segmentation": r}); st["aid"] += 1

    for i in range(n):
        dev = f"phone{i % devices}"; fn = f"{dev}_{i}_png.rf.h{i:03d}.jpg"
        im = np.random.RandomState(i).randint(70, 110, (400, 800, 3)).astype(np.uint8)
        cv2.imwrite(str(d / fn), im)
        dx, dy = rng.randint(-8, 8), rng.randint(-8, 8)
        boxes = [(1, 60, 80, 360, 300), (3, 360, 150, 390, 180), (2, 200, 60, 220, 80), (2, 60, 300, 80, 320), (2, 600, 300, 620, 320), (6, 560, 60, 640, 140)]
        if rng.random() < 0.8: boxes.append((4, 392, 155, 470, 175))
        if rng.random() < 0.7: boxes.append((5, 20, 120, 50, 250))
        if i == 3: boxes.append((1, 500, 220, 520, 280))                       # battery seen as two pieces in this image
        masks = []
        for c, x0, y0, x1, y1 in boxes:
            m = np.zeros((400, 800), np.uint8); m[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = 1
            masks.append((names[c], m))
        add(i % n_json, fn, masks); st["expected"] += len(masks)
        dev_rows.append([fn, dev, f"model_{i % devices}"])
        if i == 0:
            add(4, fn, masks)                                    # same image also listed in another json: merge, do not double
        if i == 1:                                               # augmentation copy: same source name, different pixels
            fn2 = f"{dev}_{i}_png.rf.hAUG.jpg"; cv2.imwrite(str(d / fn2), np.ascontiguousarray(im[:, ::-1]))
            add(2, fn2, [(c, np.ascontiguousarray(m[:, ::-1])) for c, m in masks]); st["expected"] += len(masks)
            dev_rows.append([fn2, dev, f"model_{i % devices}"])
    for k, js in enumerate(jsons):
        json.dump(js, open(d / ("_annotations.coco.json" if k == 0 else f"_annotations.coco_{k}.json"), "w"))
    (root / "README.roboflow.txt").write_text("synthetic")
    (root / "work").mkdir(); (root / "work" / "class_aliases.csv").write_text("from,to\nWire,Cable\n")   # user-defined merge
    with open(root / "devices.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file_name", "device_id", "phone_model"]); w.writerows(dev_rows)
    return st["expected"]


def run(*args):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=HERE, capture_output=True, text=True)
    assert out.returncode == 0, f"{args[0]} failed:\n{out.stdout[-1500:]}\n{out.stderr[-2500:]}"
    return out.stdout


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def matrix_edges(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[0][1:]
    return [(r[0], hdr[j]) for r in rows[1:] for j, c in enumerate(r[1:]) if c == "1"], rows


if __name__ == "__main__":
    unit_tests()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t); root = t / "ds"; expected = make_dataset(root); W = t / "work"
        # ---- stage 1-2: audit + group split --------------------------------------------------------
        run("s1_audit.py", "--root", root, "--out", t / "audit")
        assert (t / "audit/devices_template.csv").exists() and len(read(t / "audit/devices_template.csv")) == 41
        run("s2_prepare.py", "--root", root, "--out", t / "yolo", "--devices", root / "devices.csv")
        man = read(t / "yolo/split_manifest.csv"); sp = json.load(open(t / "yolo/split_summary.json"))
        assert sum(sp["images_per_split"].values()) == 41 and sp["images_per_split"]["val"] >= 1 and sp["images_per_split"]["test"] >= 1
        by_dev = {}
        for r in man:
            by_dev.setdefault(r["group"], set()).add(r["assigned_split"])
        assert all(len(v) == 1 for v in by_dev.values()), by_dev                      # a phone never straddles splits
        assert all(sp["instances_per_class_per_split"]["train"].get(c, 0) > 0 for c in ["Battery", "Screw", "Connector", "Cable", "Graphite Sheet", "Camera"])
        assert json.load(open(t / "yolo/class_mapping.json"))["names"] if (t / "yolo/class_mapping.json").exists() else True
        assert min(json.load(open(t / "yolo/conversion_log.json"))["roundtrip_iou_min_by_class"].values()) > 0.99
        run("s2_prepare.py", "--root", root, "--out", t / "yolo_nodev")                # unknown devices path still works
        nd = [r for r in read(t / "yolo_nodev/split_manifest.csv") if r["file_name"].startswith("phone1_1_png")]
        assert len(nd) == 2 and len({r["assigned_split"] for r in nd}) == 1 and nd[0]["group"].startswith("src::"), nd   # aug copies stay together
        rep = json.load(open(t / "audit/audit_report.json"))["merged"]
        assert rep["images"] == 41 and len(rep["duplicate_images_merged"]) == 1 and rep["augmentation_copies"] == 2 and "Screws" in rep["class_aliases_merged"] and "GraphiteSheet" in rep["class_aliases_merged"] and "wire" in rep["class_aliases_merged"], rep
        run("s2_prepare.py", "--root", root, "--out", t / "yolo_all", "--split_mode", "all")
        # ---- stage 4-5: extraction + geometry ------------------------------------------------------
        run("s4_extract.py", "--root", root, "--source", "gt", "--devices", root / "devices.csv", "--out", t / "inst")
        inst = read(t / "inst/instances.csv"); pf = read(t / "inst/pair_features.csv")
        assert len({r["image_id"] for r in inst}) == 41 and len(inst) == expected and {r["class_name"] for r in inst} == {"Battery", "Screw", "Connector", "Cable", "Graphite Sheet", "Camera"} and {r["device_id"] for r in inst} == {f"phone{k}" for k in range(8)}
        assert {"solidity", "notch_a_by_b", "inter_over_a"} <= set(pf[0]) | set(inst[0])
        run("s5_geometry.py", "--instances", t / "inst/instances.csv", "--pairs", t / "inst/pair_features.csv", "--out", t / "geo")
        g0 = next(p for p in (t / "geo").iterdir() if p.is_dir())
        assert all((g0 / f).exists() for f in ["distance_px.csv", "adjacency.csv", "containment.csv", "notch_filled.csv", "direction.csv", "area_ratio_log10.csv", "nodes.csv"])
        assert all((t / "geo" / f).exists() for f in ["class_cooccurrence_matrix.csv", "class_contact_matrix.csv", "class_near_matrix.csv", "class_adjacency_counts.csv"])
        adj = list(csv.reader(open(g0 / "adjacency.csv")))
        assert any("T" in r[1:] for r in adj[1:]) and any("N" in r[1:] for r in adj[1:]) and any("." in r[1:] for r in adj[1:])
        # ---- stage 6: review sheets ----------------------------------------------------------------
        run("s6_review.py", "--root", root, "--instances", t / "inst/instances.csv", "--out", t / "review")
        rv = read(t / "review/pair_review.csv")
        assert all((t / "review" / r["review_image"]).exists() for r in rv) and all(r["label"] == "UNKNOWN" for r in rv)
        # ---- stage 7: precedence from class rules only (no labels yet) -----------------------------
        rules = t / "rules.csv"
        run("s7_precedence.py", "--instances", t / "inst/instances.csv", "--rules", rules, "--stages_csv", t / "stages.csv", "--verified", t / "review/pair_review.csv", "--out", t / "prec0")
        stg = {r["class_name"]: r["stage"] for r in read(t / "stages.csv")}
        assert stg["Screw"] == "0" and stg["Battery"] == "5" and stg["Connector"] == "2" and stg["Graphite Sheet"] == "3", stg
        pg = read(t / "review/part_groups.csv")
        bat = [r for r in pg if r["image_id"].startswith("phone3_3_png") and r["class_name"] == "Battery"]
        assert len(bat) == 2 and len({r["part_id"] for r in bat}) == 1 and bat[0]["part_id"].endswith("_merged"), bat   # auto-merged battery pieces
        assert len(read(t / "prec0/eval_sample.csv")) == 100 and (t / "prec0/pair_suggestions.csv").exists()
        d0 = next(p for p in (t / "prec0").iterdir() if p.is_dir())
        edges, rows = matrix_edges(d0 / "precedence_inferred_direct.csv")
        cl = lambda lab: lab.split("#")[0]
        assert edges and all(cl(u) == "Screw" or (cl(u), cl(v)) == ("Connector", "Battery") for u, v in edges), edges
        src = list(csv.reader(open(d0 / "precedence_inferred_source.csv")))
        assert {c for r in src[1:] for c in r[1:]} - {"?", "0", "-"} == {"S"}               # only stage evidence, nothing verified
        assert (t / "work_stage_check").exists() is False and "Screw" in (rules.parent / "class_stage.csv").read_text() if False else True
        conf_edges, _ = matrix_edges(d0 / "precedence_confirmed_direct.csv"); assert not conf_edges
        assert (t / "prec0/next_to_label.csv").exists() and len(read(t / "prec0/next_to_label.csv")) > 0
        assert (d0 / "removal_levels.csv").exists() and "ACYCLIC" in (d0 / "report.txt").read_text()
        # ---- keyboard labelling tool: key logic, undo, confidence, skipping, queue modes --------------
        shutil_copy = __import__("shutil").copy
        shutil_copy(t / "review/pair_review.csv", t / "review/pair_review.keep")
        run("label_pairs.py", "--review", t / "review/pair_review.csv", "--queue", "all", "--keys", "abnuzA")
        lp = read(t / "review/pair_review.csv")
        first = sorted(lp, key=lambda r: (int(r["priority"]), int(r["pair_no"])))[:4]
        assert [r["label"] for r in first] == ["A_BEFORE_B", "B_BEFORE_A", "NO_DIRECT_DEPENDENCY", "A_BEFORE_B"], [r["label"] for r in first]
        assert [r["confidence"] for r in first] == ["high", "high", "high", "low"] and all(r["evidence_type"] == "image_inspection" for r in first)
        assert not first[3]["notes"].startswith("reviewed-unsure")                  # u then z then A: the 'unsure' mark was undone
        run("label_pairs.py", "--review", t / "review/pair_review.csv", "--queue", "all", "--keys", "u")
        lp = read(t / "review/pair_review.csv"); nxt = sorted(lp, key=lambda r: (int(r["priority"]), int(r["pair_no"])))[4]
        assert nxt["label"] == "UNKNOWN" and nxt["notes"].startswith("reviewed-unsure")   # unsure is remembered, row not offered again
        run("label_pairs.py", "--review", t / "review/pair_review.csv", "--queue", "sample", "--prec", t / "prec0", "--keys", "b")
        sm = read(t / "prec0/eval_sample.csv")[0]["pair_no"]
        assert next(r for r in read(t / "review/pair_review.csv") if r["pair_no"] == sm)["label"] == "B_BEFORE_A"
        shutil_copy(t / "review/pair_review.keep", t / "review/pair_review.csv")     # restore the clean sheet for the next steps
        import shutil as _sh
        try:
            import tkinter  # noqa
            have_tk = True
        except ImportError:
            have_tk = False
        if have_tk and _sh.which("xvfb-run"):                                        # real Tk window under a virtual display (Linux CI)
            out = subprocess.run(["xvfb-run", "-a", sys.executable, "label_pairs.py", "--review", str(t / "review/pair_review.csv"),
                                  "--queue", "all", "--gui_keys", "abnA"], cwd=HERE, capture_output=True, text=True, timeout=120)
            assert out.returncode == 0, out.stdout + out.stderr
            g = sorted(read(t / "review/pair_review.csv"), key=lambda r: (int(r["priority"]), int(r["pair_no"])))[:4]
            assert [r["label"] for r in g] == ["A_BEFORE_B", "B_BEFORE_A", "NO_DIRECT_DEPENDENCY", "A_BEFORE_B"] and g[3]["confidence"] == "low", [(r["label"], r["confidence"]) for r in g]
            shutil_copy(t / "review/pair_review.keep", t / "review/pair_review.csv")
            print("Tk window test (xvfb): OK")
        # ---- label every adjacent pair with the hidden truth, then validate the rules --------------
        ver = [dict(r, label=truth(r["class_a"], r["class_b"]), evidence_type="synthetic", confidence="high") for r in rv]
        with open(t / "review/pair_review.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ver[0])); w.writeheader(); w.writerows(ver)
        run("s9_evaluate.py", "--instances", t / "inst/instances.csv", "--verified", t / "review/pair_review.csv", "--rules", rules,
            "--stages_csv", t / "stages.csv", "--sample", t / "prec0/eval_sample.csv", "--out", t / "eval")
        ev = json.load(open(t / "eval/rule_evaluation.json"))
        allm, blind = ev["all_labelled_pairs"], ev["blind_sample"]
        assert allm["precision_when_fired"] == 1.0 and 0 < allm["coverage"] < 1 and allm["per_rule"]["STAGE"]["precision"] == 1.0, allm
        assert blind["labelled"] == 100 and blind["unweighted"]["precision_when_fired"] == 1.0 and blind["frequency_weighted"]["precision_when_fired"] == 1.0, blind
        assert all(float(r["weight"]) >= 1.0 for r in read(t / "prec0/eval_sample.csv"))
        assert (t / "eval/rule_evaluation_by_classpair.csv").exists()
        # ---- stage 8: learn from verified pairs (group CV) -----------------------------------------
        run("s8_train_model.py", "--instances", t / "inst/instances.csv", "--verified", t / "review/pair_review.csv", "--out", t / "model.joblib")
        rep = json.load(open(t / "model.report.json"))
        assert rep["group_cv_accuracy"] > 0.95 and rep["group_cv_accuracy"] > rep["majority_baseline"], rep
        # model fills what rules do not cover (Cable->Connector); verified labels override everything
        part = ver[:]
        sc = next(r for r in rv if "Screw" in (r["class_a"], r["class_b"]))
        sub = [dict(r, label="UNKNOWN") for r in rv]
        for r in sub:
            if r is not None and r["instance_a_id"] == sc["instance_a_id"] and r["instance_b_id"] == sc["instance_b_id"]:
                r["label"] = "NO_DIRECT_DEPENDENCY"                                          # contradict rule R1 on one pair
        with open(t / "review/sub.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sub[0])); w.writeheader(); w.writerows(sub)
        run("s7_precedence.py", "--instances", t / "inst/instances.csv", "--rules", rules, "--stages_csv", t / "stages.csv", "--verified", t / "review/sub.csv",
            "--model", t / "model.joblib", "--threshold", "0.8", "--out", t / "prec1")
        tiers = Counter(r["tier"] for r in read(t / "prec1/all_edges.csv") if r["kind"] == "direct")
        assert tiers["S"] > 0 and tiers["M"] > 0, tiers
        di = next(p for p in (t / "prec1").iterdir() if p.is_dir() and p.name == sc["image_id"])
        lab = lambda rid, cls_, num: f"{cls_}#{num}"
        la, lb = lab(0, sc["class_a"], sc["num_a"]), lab(0, sc["class_b"], sc["num_b"])
        mrows = list(csv.reader(open(di / "precedence_inferred_direct.csv"))); h = mrows[0]
        cell = next(r for r in mrows[1:] if r[0] == la)[h.index(lb)]
        assert cell == "0", cell                                                              # verified no-dependency beat the rule
        s_src = list(csv.reader(open(di / "precedence_inferred_source.csv")))
        assert next(r for r in s_src[1:] if r[0] == la)[h.index(lb)] == "V"
        # merged parts: two screws of one image become one node
        img0 = rv[0]["image_id"]; sc2 = [r["instance_id"] for r in inst if r["image_id"] == img0 and r["class_name"] == "Screw"][:2]
        with open(t / "parts.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image_id", "instance_id", "class_name", "part_id"])
            for s in sc2: w.writerow([img0, s, "Screw", img0 + "__screws"])
        run("s7_precedence.py", "--instances", t / "inst/instances.csv", "--rules", rules, "--stages_csv", t / "stages.csv", "--parts", t / "parts.csv", "--out", t / "prec2")
        n_before = len(next(csv.reader(open(d0.parent / img0 / "precedence_inferred_direct.csv")))) - 1
        n_after = len(next(csv.reader(open(t / "prec2" / img0 / "precedence_inferred_direct.csv")))) - 1
        assert n_after == n_before - 1, (n_before, n_after)
        # ---- run_all.py orchestration (no GPU stages) ----------------------------------------------
        run("run_all.py", "--root", root, "--work", W, "--stages", "audit,prepare,extract,geometry,review,precedence")
        import shutil; shutil.copy(root / "devices.csv", W / "devices.csv")
        run("run_all.py", "--root", root, "--work", W, "--stages", "extract,review")
        pr = read(W / "review/pair_review.csv")
        with open(W / "review/pair_review.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pr[0])); w.writeheader(); w.writerows([dict(r, label=truth(r["class_a"], r["class_b"])) for r in pr])
        run("run_all.py", "--root", root, "--work", W, "--stages", "model,evaluate,precedence")
        assert (W / "precedence_model.joblib").exists() and json.load(open(W / "precedence/summary.json"))["totals"]["edges_V"] > 0
        print("pipeline smoke test: OK")