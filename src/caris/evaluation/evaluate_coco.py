"""
Evaluate YOLO and Gemma object detection on the COCO val2017 dataset.

Metrics (class-level, not bounding-box IoU):
  precision = correctly detected classes / all predicted classes
  recall    = correctly detected classes / all GT classes
  f1        = harmonic mean of precision and recall
  time      = total pipeline time per image

Run:  python -m caris.evaluation.evaluate_coco [--modes yolo gemma] [--max-images 100]
Setup: make download-coco
"""

import json
import os
from collections import defaultdict
from datetime import datetime

from caris.config import COCO_DIR, DEFAULT_GEMMA_MODEL, DEFAULT_TRAINED_MODEL, OUTPUT_DIR
from caris.detection import gemma, yolo
from caris.timing import timed

DEFAULT_YOLO_MODEL = DEFAULT_TRAINED_MODEL


def _load_coco_annotations(coco_dir: str) -> dict:
    ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_path}\n"
            "Run: make download-coco"
        )
    with open(ann_path) as f:
        return json.load(f)


def _detect_yolo(model_path: str, image_path: str) -> tuple[list[str], dict]:
    return yolo.detect_with_speed(model_path, image_path, remap_person=False, use_cache=True)


def _detect_gemma(model_name: str, image_path: str) -> tuple[list[str], dict]:
    return gemma.detect_with_speed(model_name, image_path)


def _match_to_coco_classes(keywords: list[str], coco_classes: set[str]) -> set[str]:
    """Map free-form Gemma keywords to COCO class names via substring matching."""
    matched = set()
    for kw in keywords:
        kw_lower = kw.lower().strip()
        for cls in coco_classes:
            if cls in kw_lower or kw_lower in cls:
                matched.add(cls)
    return matched


def _compute_metrics(pred: set, gt: set) -> dict:
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_coco(
    coco_dir: str = COCO_DIR,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    gemma_model: str = DEFAULT_GEMMA_MODEL,
    max_images: int | None = 100,
    modes: tuple = ("yolo",),
) -> dict:
    coco_data = _load_coco_annotations(coco_dir)
    img_dir = os.path.join(coco_dir, "val2017")

    cat_map = {c["id"]: c["name"] for c in coco_data["categories"]}
    coco_classes = set(cat_map.values())

    img_to_gt: dict[int, set] = defaultdict(set)
    for ann in coco_data["annotations"]:
        img_to_gt[ann["image_id"]].add(cat_map[ann["category_id"]])

    images = [
        img for img in coco_data["images"]
        if os.path.exists(os.path.join(img_dir, img["file_name"]))
    ]
    if max_images:
        images = images[:max_images]

    print(f"Found {len(images)} images in {img_dir}")

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}

    for mode in modes:
        model_id = yolo_model if mode == "yolo" else gemma_model
        print(f"\n=== Mode: {mode} | Model: {model_id} | Images: {len(images)} ===\n")
        per_image = []

        for i, img_info in enumerate(images):
            img_path = os.path.join(img_dir, img_info["file_name"])
            gt_classes = img_to_gt.get(img_info["id"], set())
            if not gt_classes:
                continue

            with timed() as elapsed:
                try:
                    if mode == "yolo":
                        detected_raw, speed = _detect_yolo(yolo_model, img_path)
                        pred_set = set(detected_raw)
                    else:
                        detected_raw, speed = _detect_gemma(gemma_model, img_path)
                        pred_set = _match_to_coco_classes(detected_raw, coco_classes)
                except Exception as e:
                    print(f"  [WARN] {img_info['file_name']}: {e}")
                    continue
            total_time = elapsed()

            metrics = _compute_metrics(pred_set, gt_classes)
            metrics["total_pipeline_time_s"] = total_time
            metrics["model_internal_speed_s"] = speed

            entry = {
                "image": img_info["file_name"],
                "image_id": img_info["id"],
                "detected_raw": detected_raw,
                "predicted_classes": sorted(pred_set),
                "actual_classes": sorted(gt_classes),
                "matches": sorted(pred_set & gt_classes),
                "metrics": metrics,
            }
            per_image.append(entry)

            print(f"[{i+1}/{len(images)}] {img_info['file_name']}")
            if speed:
                print(f"  Model Speed (s): {speed}")
            print(f"  Predicted : {sorted(pred_set)}")
            print(f"  Actual    : {sorted(gt_classes)}")
            p, r, f = metrics["precision"], metrics["recall"], metrics["f1"]
            print(f"  Precision: {p:.2f}, Recall: {r:.2f}, F1: {f:.2f}")
            print(f"  Any match? {'YES' if entry['matches'] else 'NO'}")
            if entry["matches"]:
                print(f"  Matches   : {entry['matches']}")
            print()

        n = len(per_image)
        if n > 0:
            mean_p = sum(r["metrics"]["precision"] for r in per_image) / n
            mean_r = sum(r["metrics"]["recall"] for r in per_image) / n
            mean_f1 = sum(r["metrics"]["f1"] for r in per_image) / n
            avg_t = sum(r["metrics"]["total_pipeline_time_s"] for r in per_image) / n
        else:
            mean_p = mean_r = mean_f1 = avg_t = 0.0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model": model_id,
            "coco_dir": coco_dir,
            "total_images": len(images),
            "processed_images": n,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "mean_f1": mean_f1,
            "avg_pipeline_time_s": avg_t,
            "results": per_image,
        }
        all_results[mode] = summary

        fname = f"eval_coco_{mode}_{timestamp_str}.json"
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"--- {mode.upper()} Summary ---")
        print(f"  Processed : {n}/{len(images)}")
        print(f"  Mean Precision: {mean_p:.4f}")
        print(f"  Mean Recall   : {mean_r:.4f}")
        print(f"  Mean F1       : {mean_f1:.4f}")
        print(f"  Avg Time/img  : {avg_t:.2f}s")
        print(f"  Results saved : {fpath}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate object detection on COCO val2017")
    parser.add_argument("--coco-dir", default=COCO_DIR, help="Path to COCO dataset directory")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--max-images", type=int, default=100, help="Max images per mode (None = all 5000)")
    parser.add_argument("--modes", nargs="+", default=["yolo"], choices=["yolo", "gemma"],
                        help="Detection modes to evaluate")
    args = parser.parse_args()

    evaluate_coco(
        coco_dir=args.coco_dir,
        yolo_model=args.yolo_model,
        gemma_model=args.gemma_model,
        max_images=args.max_images,
        modes=tuple(args.modes),
    )


if __name__ == "__main__":
    main()
