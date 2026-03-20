"""
Fine-tune YOLO on the Iconclass dataset (tune_data/).

Strategy:
  Because the Iconclass dataset only provides image-level codes (no bounding
  boxes), we use a semi-supervised approach:
    1. Run the pre-trained YOLO detector on each artwork image.
    2. Map Iconclass code keywords → YOLO COCO class names.
    3. Keep only detections whose YOLO class matches an expected keyword from
       the ground-truth IC codes (discard false positives).
    4. Write the filtered detections as YOLO-format labels.
    5. Fine-tune the YOLO model on this curated pseudo-labeled dataset.

  This improves YOLO's ability to detect objects in artwork images while
  filtering out noise from initial predictions.
"""

import json
import os
import random
import shutil
from pathlib import Path

import iconclass as ic
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TUNE_DIR = "tune_data"
DEFAULT_BASE_MODEL = "yolo11n.pt"
YOLO_DATASET_DIR = os.path.join(TUNE_DIR, "yolo_dataset")
EPOCHS = 30
IMAGE_SIZE = 320
BATCH_SIZE = 16
TRAIN_FRACTION = 0.9  # within the tune set, 90% train / 10% val

# Mapping from Iconclass keywords (lowercase) → YOLO COCO class ID.
# Only keywords that clearly map to a COCO class are included.
IC_KEYWORD_TO_YOLO_CLASS = {
    "human being": 0, "man": 0, "woman": 0, "child": 0, "boy": 0,
    "girl": 0, "Christ": 0, "saint": 0, "soldier": 0, "king": 0,
    "queen": 0, "angel": 0, "monk": 0, "priest": 0, "knight": 0,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18,
    "cow": 19, "elephant": 20, "bear": 21,
    "boat": 8, "ship": 8,
    "book": 73, "clock": 74, "vase": 75, "scissors": 76,
    "knife": 43, "cup": 41, "chair": 56, "bottle": 39, "bowl": 45,
    "apple": 47, "umbrella": 25, "cake": 55,
}

# Reverse: YOLO class ID → set of IC keywords that map to it
YOLO_CLASS_TO_IC_KEYWORDS: dict[int, set[str]] = {}
for _kw, _cls_id in IC_KEYWORD_TO_YOLO_CLASS.items():
    YOLO_CLASS_TO_IC_KEYWORDS.setdefault(_cls_id, set()).add(_kw.lower())


# ---------------------------------------------------------------------------
# Data helpers
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


def load_tune_data(tune_dir: str = TUNE_DIR) -> dict:
    path = os.path.join(tune_dir, "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tune data not found at {path}")
    with open(path) as f:
        return json.load(f)


def get_expected_yolo_classes(ic_codes: list[str]) -> set[int]:
    """Given a list of Iconclass codes, return the set of YOLO class IDs
    that are expected to appear in the image (based on IC keyword mapping)."""
    tree = ic.init()
    expected: set[int] = set()
    for code_str in ic_codes:
        try:
            notation = tree[code_str]
            keywords = [kw.lower() for kw in notation.keywords()]
            for kw in keywords:
                if kw in IC_KEYWORD_TO_YOLO_CLASS:
                    expected.add(IC_KEYWORD_TO_YOLO_CLASS[kw])
        except Exception:
            continue
    return expected


def build_keyword_to_images_map(data: dict) -> dict[str, list[str]]:
    """Map each IC keyword (that has a YOLO equivalent) to list of image names."""
    tree = ic.init()
    kw_images: dict[str, list[str]] = {}
    for img_name, entry in data.items():
        for code_str in _extract_ic_codes(entry):
            try:
                notation = tree[code_str]
                for kw in notation.keywords():
                    kw_lower = kw.lower()
                    if kw_lower in IC_KEYWORD_TO_YOLO_CLASS:
                        kw_images.setdefault(kw_lower, []).append(img_name)
            except Exception:
                continue
    return kw_images


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    tune_dir: str,
    data: dict,
    base_model: str = DEFAULT_BASE_MODEL,
    confidence_threshold: float = 0.25,
    max_images: int | None = None,
) -> tuple[list[str], list[str]]:
    """
    Run YOLO on tune images, keep only detections consistent with IC codes,
    and write YOLO-format label files.

    Returns (train_images, val_images) — lists of image paths.
    """
    model = YOLO(base_model)
    image_dir = os.path.join(tune_dir, "images")

    # Prepare YOLO dataset directory structure
    for split in ("train", "val"):
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, "labels", split), exist_ok=True)

    keys = list(data.keys())
    if max_images is not None:
        keys = keys[:max_images]

    # Deterministic train/val split within tune data
    random.seed(42)
    random.shuffle(keys)
    n_train = int(len(keys) * TRAIN_FRACTION)
    train_keys = set(keys[:n_train])

    usable_train, usable_val = [], []
    n_labeled = 0

    print(f"Generating pseudo-labels for {len(keys)} images ...")
    for idx, img_name in enumerate(keys):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        ic_codes = _extract_ic_codes(data[img_name])
        expected_classes = get_expected_yolo_classes(ic_codes)
        if not expected_classes:
            continue  # no YOLO-mappable IC codes for this image

        # Run YOLO detection
        try:
            results = model(img_path, verbose=False)
        except Exception:
            continue

        # Filter detections: keep only those matching expected IC keywords
        label_lines = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                if cls_id in expected_classes and conf >= confidence_threshold:
                    # YOLO format: class x_center y_center width height (normalized)
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    xc = (x1 + x2) / 2
                    yc = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not label_lines:
            continue  # no valid detections for this image

        # Assign to train or val
        split = "train" if img_name in train_keys else "val"

        # Copy image
        dest_img = os.path.join(YOLO_DATASET_DIR, "images", split, img_name)
        if not os.path.exists(dest_img):
            shutil.copy2(img_path, dest_img)

        # Write label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        dest_label = os.path.join(YOLO_DATASET_DIR, "labels", split, label_name)
        with open(dest_label, "w") as f:
            f.write("\n".join(label_lines) + "\n")

        if split == "train":
            usable_train.append(dest_img)
        else:
            usable_val.append(dest_img)
        n_labeled += 1

        if (idx + 1) % 5000 == 0:
            print(f"  ... processed {idx + 1}/{len(keys)} images, {n_labeled} labeled")

    print(f"Pseudo-labeling complete: {len(usable_train)} train, {len(usable_val)} val images")
    return usable_train, usable_val


# ---------------------------------------------------------------------------
# Dataset YAML
# ---------------------------------------------------------------------------

def write_dataset_yaml(dataset_dir: str = YOLO_DATASET_DIR) -> str:
    """Write a YOLO dataset.yaml for training."""
    # Collect all class IDs actually used in labels
    used_classes: set[int] = set()
    for split in ("train", "val"):
        label_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(label_dir, fname)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            used_classes.add(int(parts[0]))

    # Use the standard COCO names for each class
    coco_names = {
        0: "person", 8: "boat", 14: "bird", 15: "cat", 16: "dog",
        17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        25: "umbrella", 39: "bottle", 41: "cup", 43: "knife", 45: "bowl",
        47: "apple", 55: "cake", 56: "chair", 73: "book", 74: "clock",
        75: "vase", 76: "scissors",
    }

    names_dict = {cid: coco_names.get(cid, f"class_{cid}") for cid in sorted(used_classes)}

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    abs_dir = os.path.abspath(dataset_dir)

    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write(f"nc: {len(names_dict)}\n")
        f.write("names:\n")
        for cid, name in names_dict.items():
            f.write(f"  {cid}: {name}\n")

    print(f"Wrote dataset config: {yaml_path}")
    print(f"  Classes used: {names_dict}")
    return yaml_path


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def fine_tune(
    tune_dir: str = TUNE_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMAGE_SIZE,
    batch: int = BATCH_SIZE,
    max_images: int | None = None,
):
    """Full fine-tuning pipeline: pseudo-label → write YAML → train YOLO."""
    data = load_tune_data(tune_dir)
    print(f"Loaded {len(data)} images from {tune_dir}")

    # Step 1: generate pseudo-labels
    train_imgs, val_imgs = generate_pseudo_labels(
        tune_dir, data, base_model,
        max_images=max_images,
    )

    if not train_imgs:
        print("No usable training images found. Aborting fine-tune.")
        return

    # Step 2: write dataset YAML
    yaml_path = write_dataset_yaml()

    # Step 3: train
    print(f"\nStarting YOLO fine-tuning: {epochs} epochs, imgsz={imgsz}, batch={batch}")
    model = YOLO(base_model)
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/fine_tune",
        name="iconclass",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    print("\nFine-tuning complete. Weights saved under runs/fine_tune/iconclass/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Iconclass dataset")
    parser.add_argument("--tune-dir", default=TUNE_DIR, help="Path to tune dataset directory")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base YOLO model weights")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Image size for training")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to use (for quick tests)")
    args = parser.parse_args()

    fine_tune(
        tune_dir=args.tune_dir,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        max_images=args.max_images,
    )
