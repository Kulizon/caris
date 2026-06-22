import json
import os
from datetime import datetime

from caris.config import DEFAULT_TRAINED_MODEL, EVAL_DIR, OUTPUT_DIR
from caris.pipeline import get_iconclass_codes
from caris.timing import timed

DEFAULT_MODEL = DEFAULT_TRAINED_MODEL
# DEFAULT_MODEL = "gemma3:12b"


def _extract_ic_codes(entry) -> list[str]:
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return entry.get("IC", [])
    return []


def load_eval_data(eval_dir: str = EVAL_DIR) -> dict:
    path = os.path.join(eval_dir, "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval data not found at {path}")
    with open(path) as f:
        return json.load(f)

def evaluate(
    eval_dir: str = EVAL_DIR,
    trained_model: str = DEFAULT_MODEL,
    mapping_mode: str = "auto",
    max_images: int | None = None,
) -> dict:
    data = load_eval_data(eval_dir)
    image_dir = os.path.join(eval_dir, "images")

    keys = list(data.keys())
    if max_images is not None:
        keys = keys[:max_images]

    results = []
    total_images = len(keys)
    processed_images = 0
    correct_images = 0

    # Prepare output path early for incremental saving
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    model_basename = os.path.basename(trained_model).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_basename}_{mapping_mode}_{timestamp_str}.json"
    output_path = os.path.join(output_dir, filename)

    print(f"Evaluating {total_images} images from {eval_dir} ...")
    for idx, img_name in enumerate(keys):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        gt_codes = sorted(list(set(_extract_ic_codes(data[img_name]))))
        if not gt_codes:
            continue

        processed_images += 1
        with timed() as elapsed:
            try:
                pred_codes, detected, model_speed = get_iconclass_codes(
                    trained_model=trained_model,
                    image_path=img_path,
                    mapping_mode=mapping_mode
                )
                pred_codes_set = set(pred_codes)
            except Exception as e:
                print(f"  [WARN] {img_name}: {e}")
                continue
        total_pipeline_time = elapsed()

        pred_codes = sorted(list(pred_codes_set))
        gt_set = set(gt_codes)
        pred_set = set(pred_codes)
        matches = sorted(list(pred_set & gt_set))

        # Calculate metrics for this image
        precision = len(matches) / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = len(matches) / len(gt_set) if len(gt_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        is_correct = len(matches) > 0
        if is_correct:
            correct_images += 1

        res_entry = {
            "image": img_name,
            "detected_objects": detected,
            "predicted_codes": pred_codes,
            "actual_codes": gt_codes,
            "matches": matches,
            "is_correct": is_correct,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total_pipeline_time_s": total_pipeline_time,
                "model_internal_speed_s": model_speed
            }
        }
        results.append(res_entry)

        print(f"Image    : {img_name} (Total: {total_pipeline_time:.4f}s)")
        if model_speed:
            print(f"  Model Speed (s): {model_speed}")
        print(f"  Predicted: {pred_codes if pred_codes else '(none)'}")
        print(f"  Actual   : {gt_codes}")
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        print(f"  Any actual in predicted? {'YES' if matches else 'NO'}")
        if matches:
            print(f"  Matches   : {matches}")
        print()

        # Incremental save every 5 images
        if (processed_images > 0) and (processed_images % 5 == 0 or idx == len(keys) - 1):
            temp_summary = {
                "timestamp": datetime.now().isoformat(),
                "model": trained_model,
                "mapping_mode": mapping_mode,
                "eval_dir": eval_dir,
                "total_images": total_images,
                "processed_images": processed_images,
                "correct_images": correct_images,
                "accuracy": (correct_images / processed_images) if processed_images > 0 else 0,
                "results": results
            }
            with open(output_path, "w") as f:
                json.dump(temp_summary, f, indent=4)

        if (idx + 1) % 100 == 0:
            pct = 100.0 * (idx + 1) / total_images
            print(f"  ... {idx + 1}/{total_images} ({pct:.1f}%)")

    # Final summary with full metrics
    mean_precision = sum(r["metrics"]["precision"] for r in results) / processed_images if processed_images > 0 else 0
    mean_recall = sum(r["metrics"]["recall"] for r in results) / processed_images if processed_images > 0 else 0
    mean_f1 = sum(r["metrics"]["f1"] for r in results) / processed_images if processed_images > 0 else 0
    total_time = sum(r["metrics"]["total_pipeline_time_s"] for r in results)
    avg_time = total_time / processed_images if processed_images > 0 else 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": trained_model,
        "mapping_mode": mapping_mode,
        "eval_dir": eval_dir,
        "total_images": total_images,
        "processed_images": processed_images,
        "correct_images": correct_images,
        "accuracy": (correct_images / processed_images) if processed_images > 0 else 0,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "total_evaluation_time": total_time,
        "avg_pipeline_time": avg_time,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Evaluation finished. Results saved to {output_path}")
    print(f"Accuracy (Any match): {summary['accuracy']:.2%} ({correct_images}/{processed_images})")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall   : {mean_recall:.4f}")
    print(f"Mean F1       : {mean_f1:.4f}")
    print(f"Avg Time/Img  : {avg_time:.2f}s")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CARIS Iconclass detection")
    parser.add_argument("--eval-dir", default=EVAL_DIR, help="Path to eval dataset directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model weights")
    parser.add_argument("--mapping", default="auto", choices=["auto", "structural", "semantic"], help="Mapping mode")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate (for quick tests)")
    args = parser.parse_args()

    evaluate(
        eval_dir=args.eval_dir,
        trained_model=args.model,
        mapping_mode=args.mapping,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
