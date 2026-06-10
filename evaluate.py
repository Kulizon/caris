import json
import os
import time
from datetime import datetime

from main import find_iconclass_tags
from utils import detect_objects_in_image
from classification_utils import get_iconclass_codes_gemma

DEFAULT_MODEL = "models/yolo26n.pt"
# DEFAULT_MODEL = "gemma3:12b"
EVAL_DIR = "eval_data"

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


def predict_iconclass_codes(
    image_path: str,
    trained_model: str = DEFAULT_MODEL,
) -> tuple[set[str], list[str], dict]:
    # 1. Detection
    detected, speed_metrics = detect_objects_in_image(trained_model, image_path)
    if len(detected) == 0:
        return set(), detected, speed_metrics
    
    # 2. Tag finding (passing already detected objects)
    iconclass_tags = find_iconclass_tags(
        image_name=image_path,
        iconclass_branch_to_start_from="",
        trained_model=trained_model,
        detected_objects=detected
    )
    flat: set[str] = set()
    for code_set in iconclass_tags:
        for code in code_set:
            flat.add(str(code))
    return flat, detected, speed_metrics


def predict_iconclass_codes_gemma(
    image_path: str,
    trained_model: str,
) -> tuple[set[str], list[str], dict]:
    codes, classified_objects, speed_metrics = get_iconclass_codes_gemma(trained_model, image_path)
    flat: set[str] = set()
    for code in codes:
        flat.add(str(code))
    return flat, classified_objects, speed_metrics


def evaluate(
    eval_dir: str = EVAL_DIR,
    trained_model: str = DEFAULT_MODEL,
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
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_basename = os.path.basename(trained_model).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_basename}_{timestamp_str}.json"
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
        start_time = time.time()
        try:
            if "gemma" in trained_model.lower() or "ram" in trained_model.lower():
                pred_codes_set, detected, model_speed = predict_iconclass_codes_gemma(img_path, trained_model)
            else:
                pred_codes_set, detected, model_speed = predict_iconclass_codes(img_path, trained_model)
        except Exception as e:
            print(f"  [WARN] {img_name}: {e}")
            continue
        total_pipeline_time = time.time() - start_time

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CARIS Iconclass detection")
    parser.add_argument("--eval-dir", default=EVAL_DIR, help="Path to eval dataset directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model weights")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate (for quick tests)")
    args = parser.parse_args()

    evaluate(
        eval_dir=args.eval_dir,
        trained_model=args.model,
        max_images=args.max_images,
    )
