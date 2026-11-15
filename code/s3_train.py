#!/usr/bin/env python3
"""
Step 3: fine-tune pretrained YOLO11n-seg on the prepared dataset (laptop / 6 GB VRAM).

    python s3_train.py --data data\\processed\\yolo_seg\\dataset.yaml
    python s3_train.py --data ... --smoke          # 1 epoch, imgsz 640, to check the setup

Default imgsz is 1280 because the median screw is ~28x28 px on a 2000x1000 image
(about 9x9 px at 640). If you hit CUDA out-of-memory, lower --imgsz or use --batch 1.
With 8 train / 2 val / 1 test images, metrics are NOT meaningful - this is a pipeline test.
"""
import argparse, json, platform, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="checkpoints")
    ap.add_argument("--name", default="seg_baseline")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.imgsz, a.name = 1, 640, a.name + "_smoke"

    import torch, ultralytics
    from ultralytics import YOLO
    random.seed(a.seed); torch.manual_seed(a.seed)

    Path("logs").mkdir(exist_ok=True)
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available(),
           "cuda": torch.version.cuda, "args": {k: str(v) for k, v in vars(a).items()}}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 2))
    json.dump(env, open(f"logs/env_{a.name}.json", "w"), indent=2)
    print(json.dumps(env, indent=2))

    project = Path(a.project).resolve()   # absolute, so Ultralytics does not nest it under runs/segment
    model = YOLO(a.model)
    model.train(data=str(a.data), imgsz=a.imgsz, batch=a.batch, epochs=a.epochs,
                patience=a.patience, seed=a.seed, workers=2, amp=True, deterministic=True,
                device=a.device, project=str(project), name=a.name, exist_ok=True)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    results = {}
    for split in ("val", "test"):
        try:
            m = YOLO(str(best)).val(data=str(a.data), split=split, imgsz=a.imgsz, batch=a.batch,
                                    device=a.device, project=str(project), name=f"{a.name}_{split}", exist_ok=True)
            results[split] = {"mask_map50": float(m.seg.map50), "mask_map50_95": float(m.seg.map)}
        except Exception as e:  # e.g. split has no usable labels
            results[split] = {"error": str(e)}
    results["NOTE"] = "Small dataset: val/test have only a handful of images, so per-class numbers (esp. rare classes) are noisy. Report them with the image counts."
    json.dump(results, open(f"logs/metrics_{a.name}.json", "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("best weights:", best)


if __name__ == "__main__":
    main()
