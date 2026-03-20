"""
Evaluate the CARIS Iconclass detection pipeline.

Takes the evaluation dataset (eval_data/) and produces a short report
comparing YOLO-based Iconclass code predictions against ground-truth codes.

Metrics reported:
  - Per-image precision, recall, F1 (averaged over the eval set)
  - Hierarchical partial-match accuracy (shared-prefix credit)
  - Breakdown by top-level Iconclass branch
"""

import json
import os
import sys
from collections import defaultdict

import iconclass as ic

from classification_utils import (
    search_for_equal_tags_in_subtree,
    search_for_subset_of_tags_in_subtree,
)
from utils import detect_objects_in_image

DEFAULT_MODEL = "yolo11n.pt"
EVAL_DIR = "eval_data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ic_codes(entry) -> list[str]:
    """Extract IC codes from a data entry, handling both formats:
    - list of codes:  ["31A71", "41D2321"]              (zip's data.json)
    - dict with IC key: {"IC": [...], "CAPTION": [...]}  (test_data.json)
    """
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        return entry.get("IC", [])
    return []


def load_eval_data(eval_dir: str = EVAL_DIR) -> dict:
    """Load data.json from the evaluation directory."""
    path = os.path.join(eval_dir, "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval data not found at {path}")
    with open(path) as f:
        return json.load(f)


def predict_iconclass_codes(
    image_path: str,
    trained_model: str = DEFAULT_MODEL,
    search_individually: str = "ALWAYS",
) -> set[str]:
    """Run the CARIS classification pipeline on a single image and return predicted IC codes."""
    tree = ic.init()
    root = tree[""]

    detected_objects = detect_objects_in_image(trained_model, image_path)
    # Apply the same person→human being mapping as in utils.py
    # (already handled there, but just in case)

    result_codes: list[set] = []

    match = search_for_equal_tags_in_subtree(root, detected_objects)
    if match:
        result_codes.append(set(str(c) for c in match))
    else:
        match = search_for_subset_of_tags_in_subtree(root, detected_objects)
        if match:
            result_codes.append(set(str(c) for c in match))

    if search_individually == "ALWAYS" or (
        search_individually == "IF_NONE_FOUND" and not result_codes
    ):
        for obj in detected_objects:
            m = search_for_equal_tags_in_subtree(root, [obj])
            if m:
                result_codes.append(set(str(c) for c in m))
            else:
                m = search_for_subset_of_tags_in_subtree(root, [obj])
                if m:
                    result_codes.append(set(str(c) for c in m))

    all_codes: set[str] = set()
    for s in result_codes:
        all_codes |= s
    return all_codes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _strip_qualifier(code: str) -> str:
    """Remove Iconclass bracket qualifiers for comparison, e.g. '25F24(+78)' -> '25F24'."""
    return code.split("(")[0].strip()


def _top_branch(code: str) -> str:
    """Return the single-digit top-level Iconclass branch."""
    stripped = _strip_qualifier(code)
    return stripped[0] if stripped else "?"


def prefix_similarity(a: str, b: str) -> float:
    """Score how much two IC codes share a common prefix (0..1)."""
    a, b = _strip_qualifier(a), _strip_qualifier(b)
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    common = 0
    for ca, cb in zip(a, b):
        if ca == cb:
            common += 1
        else:
            break
    return common / max_len


def best_prefix_score(pred_code: str, gt_codes: set[str]) -> float:
    """Best prefix similarity of a predicted code against all ground-truth codes."""
    if not gt_codes:
        return 0.0
    return max(prefix_similarity(pred_code, g) for g in gt_codes)


def compute_metrics(predicted: set[str], ground_truth: set[str]) -> dict:
    """Compute precision, recall, F1 (exact & partial) for a single image."""
    pred_stripped = {_strip_qualifier(c) for c in predicted}
    gt_stripped = {_strip_qualifier(c) for c in ground_truth}

    # Exact match
    tp = len(pred_stripped & gt_stripped)
    precision = tp / len(pred_stripped) if pred_stripped else 0.0
    recall = tp / len(gt_stripped) if gt_stripped else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Partial (prefix) match: average best-prefix-score for each predicted code
    partial_scores = [best_prefix_score(p, gt_stripped) for p in pred_stripped] if pred_stripped else [0.0]
    avg_partial = sum(partial_scores) / len(partial_scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "partial_match": avg_partial,
        "n_predicted": len(pred_stripped),
        "n_ground_truth": len(gt_stripped),
        "n_exact_matches": tp,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    eval_dir: str = EVAL_DIR,
    trained_model: str = DEFAULT_MODEL,
    max_images: int | None = None,
) -> dict:
    """
    Run full evaluation and return summary dict.

    Parameters
    ----------
    eval_dir : str
        Directory containing data.json and images/.
    trained_model : str
        Path to the YOLO model weights.
    max_images : int or None
        Cap the number of images evaluated (useful for quick checks).
    """
    data = load_eval_data(eval_dir)
    image_dir = os.path.join(eval_dir, "images")

    keys = list(data.keys())
    if max_images is not None:
        keys = keys[:max_images]

    all_metrics: list[dict] = []
    branch_metrics: dict[str, list[dict]] = defaultdict(list)
    n_skipped = 0

    print(f"Evaluating {len(keys)} images from {eval_dir} ...")
    for idx, img_name in enumerate(keys):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            n_skipped += 1
            continue

        gt_codes = set(_extract_ic_codes(data[img_name]))
        if not gt_codes:
            n_skipped += 1
            continue

        try:
            pred_codes = predict_iconclass_codes(img_path, trained_model, search_individually="IF_NONE_FOUND")
        except Exception as e:
            print(f"  [WARN] {img_name}: {e}")
            n_skipped += 1
            continue

        m = compute_metrics(pred_codes, gt_codes)
        all_metrics.append(m)

        # Track per-branch
        for code in gt_codes:
            branch = _top_branch(code)
            branch_metrics[branch].append(m)

        if (idx + 1) % 200 == 0:
            print(f"  ... {idx + 1}/{len(keys)}")

    # --- Aggregate ---------------------------------------------------------
    n = len(all_metrics)
    if n == 0:
        print("No images evaluated.")
        return {}

    avg = lambda key: sum(m[key] for m in all_metrics) / n

    summary = {
        "images_evaluated": n,
        "images_skipped": n_skipped,
        "avg_precision": avg("precision"),
        "avg_recall": avg("recall"),
        "avg_f1": avg("f1"),
        "avg_partial_match": avg("partial_match"),
        "avg_predicted_codes": avg("n_predicted"),
        "avg_gt_codes": avg("n_ground_truth"),
    }

    # Per-branch summary
    branch_summary = {}
    for branch, metrics in sorted(branch_metrics.items()):
        bn = len(metrics)
        branch_summary[branch] = {
            "count": bn,
            "avg_f1": sum(m["f1"] for m in metrics) / bn,
            "avg_partial": sum(m["partial_match"] for m in metrics) / bn,
        }
    summary["per_branch"] = branch_summary

    return summary


def print_report(summary: dict):
    """Print a concise human-readable evaluation report."""
    if not summary:
        print("No results to report.")
        return

    print()
    print("=" * 60)
    print("  CARIS Iconclass Detection — Evaluation Report")
    print("=" * 60)
    print(f"  Images evaluated : {summary['images_evaluated']}")
    print(f"  Images skipped   : {summary['images_skipped']}")
    print(f"  Avg GT codes/img : {summary['avg_gt_codes']:.1f}")
    print(f"  Avg pred codes   : {summary['avg_predicted_codes']:.1f}")
    print("-" * 60)
    print(f"  Exact precision  : {summary['avg_precision']:.4f}")
    print(f"  Exact recall     : {summary['avg_recall']:.4f}")
    print(f"  Exact F1         : {summary['avg_f1']:.4f}")
    print(f"  Partial match    : {summary['avg_partial_match']:.4f}")
    print("-" * 60)
    print("  Per top-level Iconclass branch:")
    for branch, bdata in summary.get("per_branch", {}).items():
        print(f"    Branch {branch}: F1={bdata['avg_f1']:.4f}  "
              f"partial={bdata['avg_partial']:.4f}  (n={bdata['count']})")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CARIS Iconclass detection")
    parser.add_argument("--eval-dir", default=EVAL_DIR, help="Path to eval dataset directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO model weights")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to evaluate (for quick tests)")
    args = parser.parse_args()

    summary = evaluate(
        eval_dir=args.eval_dir,
        trained_model=args.model,
        max_images=args.max_images,
    )
    print_report(summary)
