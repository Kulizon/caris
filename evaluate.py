

import json
import os

from main import find_iconclass_tags
from utils import detect_objects_in_image

DEFAULT_MODEL = "models/yolo26n.pt"
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
) -> tuple[set[str], list[str]]:
    detected = detect_objects_in_image(trained_model, image_path)
    if len(detected) == 0:
        return set(), detected
    iconclass_tags = find_iconclass_tags(
        image_name=image_path,
        iconclass_branch_to_start_from="",
        trained_model=trained_model,
    )
    flat: set[str] = set()
    for code_set in iconclass_tags:
        for code in code_set:
            flat.add(str(code))
    return flat, detected


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

    print(f"Evaluating {len(keys)} images from {eval_dir} ...")
    for idx, img_name in enumerate(keys):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        gt_codes = set(_extract_ic_codes(data[img_name]))
        if not gt_codes:
            continue

        try:
            pred_codes, detected = predict_iconclass_codes(img_path, trained_model)
        except Exception as e:
            print(f"  [WARN] {img_name}: {e}")
            continue

        matches = sorted(set(pred_codes) & gt_codes)

        print(f"Image    : {img_name}")
        print(f"  YOLO   : {detected}")
        print(f"  Predicted: {sorted(pred_codes) if pred_codes else '(none)'}")
        print(f"  Actual   : {sorted(gt_codes)}")
        print(f"  Any actual in predicted? {'YES' if matches else 'NO'}")
        if matches:
            print(f"  Matches   : {matches}")
        print()
        if (idx + 1) % 100 == 0:
            pct = 100.0 * (idx + 1) / len(keys)
            print(f"  ... {idx + 1}/{len(keys)} ({pct:.1f}%)")
    return None


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
