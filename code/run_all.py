#!/usr/bin/env python3
"""
One entry point for the whole project. Run from the folder that holds these scripts.

    python run_all.py --root D:\\precedence_matrix --stages audit            # first look + devices template
    python run_all.py --root D:\\precedence_matrix --stages prepare,train    # YOLO dataset + fine-tune (GPU)
    python run_all.py --root D:\\precedence_matrix --stages extract,geometry,review,precedence
    python run_all.py --root D:\\precedence_matrix --stages evaluate        # after you have labelled pairs (rule precision)
    python run_all.py --root D:\\precedence_matrix --stages model,evaluate,precedence   # learn from your labels
    python run_all.py --images D:\\new_phones --weights work\\checkpoints\\seg\\weights\\best.pt --stages infer

Everything is written under --work (default <root>\\work). Stages can be re-run any time.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT = "audit,prepare,extract,geometry,review,precedence"


def run(*args):
    cmd = [sys.executable, *map(str, args)]
    print("\n>>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode:
        sys.exit(f"stage failed (exit {r.returncode}) - fix the error above and rerun")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="folder with the Roboflow COCO export")
    ap.add_argument("--work", type=Path)
    ap.add_argument("--stages", default=DEFAULT, help="comma list of: audit prepare train extract geometry review label precedence model evaluate infer")
    ap.add_argument("--images", type=Path, help="folder of new images for the infer stage")
    ap.add_argument("--weights", type=Path, help="segmentation weights (best.pt) for infer")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--split_mode", default="group")
    ap.add_argument("--queue", default="sample", help="label stage: sample (blind, do first) | next | all")
    ap.add_argument("--regroup", action="store_true", help="review stage: rebuild part_groups.csv (auto-merges split Battery pieces)")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.root is None and "infer" not in stages:
        ap.error("--root is required")
    work = (a.work or (a.root / "work" if a.root else Path("work"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    devices = work / "devices.csv"
    dev_args = ["--devices", devices] if devices.exists() else []
    inst, review, prec = work / "instances", work / "review", work / "precedence"
    rules, stages_csv, model = work / "class_precedence.csv", work / "class_stage.csv", work / "precedence_model.joblib"

    for s in stages:
        if s == "audit":
            run("s1_audit.py", "--root", a.root, "--out", work / "audit")
            if not devices.exists():
                shutil.copy(work / "audit" / "devices_template.csv", devices)
                print(f"\nFILL IN {devices}: put the same device_id on images of the same phone (and phone_model if you know it).")
        elif s == "prepare":
            run("s2_prepare.py", "--root", a.root, "--out", work / "yolo_seg", "--split_mode", a.split_mode, *dev_args)
        elif s == "train":
            run("s3_train.py", "--data", work / "yolo_seg" / "dataset.yaml", "--epochs", a.epochs, "--imgsz", a.imgsz,
                "--patience", 100, "--project", work / "checkpoints", "--name", "seg")
        elif s == "extract":
            run("s4_extract.py", "--root", a.root, "--source", "gt", "--out", inst, *dev_args)
        elif s == "geometry":
            run("s5_geometry.py", "--instances", inst / "instances.csv", "--pairs", inst / "pair_features.csv", "--out", work / "geometry")
        elif s == "review":
            run("s6_review.py", "--root", a.root, "--instances", inst / "instances.csv", "--out", review, *(["--regroup"] if a.regroup else []))
        elif s == "label":
            run("label_pairs.py", "--review", review / "pair_review.csv", "--queue", a.queue, "--prec", prec)
        elif s == "precedence":
            args = ["s7_precedence.py", "--instances", inst / "instances.csv", "--rules", rules, "--stages_csv", stages_csv, "--out", prec]
            if (review / "pair_review.csv").exists():
                args += ["--verified", review / "pair_review.csv"]
            if (review / "part_groups.csv").exists():
                args += ["--parts", review / "part_groups.csv"]
            if model.exists():
                args += ["--model", model]
            run(*args)
        elif s == "model":
            run("s8_train_model.py", "--instances", inst / "instances.csv", "--verified", review / "pair_review.csv", "--out", model)
        elif s == "evaluate":
            args = ["s9_evaluate.py", "--instances", inst / "instances.csv", "--verified", review / "pair_review.csv",
                    "--rules", rules, "--stages_csv", stages_csv, "--out", prec, "--calibrate"]
            if (prec / "eval_sample.csv").exists():
                args += ["--sample", prec / "eval_sample.csv"]
            run(*args)
        elif s == "infer":
            if not (a.images and a.weights):
                ap.error("infer needs --images and --weights")
            out = work / "infer"
            run("s4_extract.py", "--images", a.images, "--weights", a.weights, "--out", out / "instances", *dev_args)
            run("s5_geometry.py", "--instances", out / "instances" / "instances.csv", "--pairs", out / "instances" / "pair_features.csv", "--out", out / "geometry")
            args = ["s7_precedence.py", "--instances", out / "instances" / "instances.csv", "--rules", rules, "--stages_csv", stages_csv, "--out", out / "precedence"]
            if model.exists():
                args += ["--model", model]
            run(*args)
        else:
            ap.error(f"unknown stage {s}")
    print("\nDone:", ", ".join(stages), "->", work)


if __name__ == "__main__":
    main()
