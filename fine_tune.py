import json
import os
import shutil
import random
import requests
import iconclass
from collections import Counter

TESTSET_IMAGE_URL = "https://iconclass.org/testset/images/"
DATA_JSON = "test_data.json"
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iconclass_dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
VAL_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 30
MAX_SAMPLES_PER_CLASS = 500
FINETUNE_EPOCHS = 50
FINETUNE_IMGSZ = 224
FINETUNE_BATCH = 32
BASE_MODEL = "yolo11n.pt"
FINETUNED_MODEL_NAME = "yolo11n-iconclass"

TARGET_KEYWORDS = {
    "dog", "cat", "horse", "bird", "sheep", "cow", "bear",
    "deer", "lion", "snake", "fish", "monkey", "rabbit",
    "goat", "donkey", "camel", "pig", "wolf", "fox", "dragon",
    "swan", "dove", "owl", "stag",
    "sword", "shield", "crown", "cross", "book", "cup", "candle", "key",
    "mirror", "ring", "lamp", "bell", "wheel", "anchor", "arrow",
    "harp", "drum", "trumpet", "flag", "skull", "globe", "torch",
    "flower", "tree", "sun", "moon", "star", "mountain", "river",
    "castle", "church", "tower", "bridge", "ship", "boat",
    "man", "woman", "child", "angel", "soldier",
    "sailing-ship", "cupid", "shepherd", "scroll", "staff",
    "vase", "table", "glass", "fruit",
}


def load_test_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_keywords_for_code(ic, code_str):
    try:
        node = ic[code_str]
        return [kw.lower() for kw in node.keywords()]
    except Exception:
        return []


def build_keyword_to_images_map(data):
    ic = iconclass.init()
    keyword_images = {}
    for image_name, entry in data.items():
        codes = entry.get("IC", [])
        image_keywords = set()
        for code in codes:
            kws = get_keywords_for_code(ic, code)
            for kw in kws:
                if kw in TARGET_KEYWORDS:
                    image_keywords.add(kw)
        for kw in image_keywords:
            keyword_images.setdefault(kw, []).append(image_name)
    return keyword_images


def download_image(image_name, dest_path):
    if os.path.exists(dest_path):
        return True
    url = TESTSET_IMAGE_URL + image_name
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False


def prepare_dataset(keyword_images):
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
        if os.path.exists(DATASET_DIR):
            os.system(f"rm -rf '{DATASET_DIR}'")

    stats = {}
    for keyword, images in sorted(keyword_images.items()):
        if len(images) < MIN_SAMPLES_PER_CLASS:
            print(f"  Skipping '{keyword}': only {len(images)} images (min={MIN_SAMPLES_PER_CLASS})")
            continue

        if len(images) > MAX_SAMPLES_PER_CLASS:
            images = random.sample(images, MAX_SAMPLES_PER_CLASS)

        random.shuffle(images)
        split_idx = max(1, int(len(images) * (1 - VAL_SPLIT)))
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        class_name = keyword.replace(" ", "_")
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        downloaded_train = 0
        for img in train_images:
            dest = os.path.join(train_class_dir, img)
            if download_image(img, dest):
                downloaded_train += 1

        downloaded_val = 0
        for img in val_images:
            dest = os.path.join(val_class_dir, img)
            if download_image(img, dest):
                downloaded_val += 1

        stats[keyword] = {"train": downloaded_train, "val": downloaded_val}
        print(f"  '{keyword}': {downloaded_train} train, {downloaded_val} val")

    return stats


def finetune(dataset_path):
    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=os.path.abspath(dataset_path),
        epochs=FINETUNE_EPOCHS,
        imgsz=FINETUNE_IMGSZ,
        batch=FINETUNE_BATCH,
        name=FINETUNED_MODEL_NAME,
        pretrained=True,
    )
    return results


if __name__ == "__main__":
    print("Step 1: Loading Iconclass test data...")
    data = load_test_data(DATA_JSON)
    print(f"  Loaded {len(data)} images with Iconclass codes.")

    print("\nStep 2: Mapping Iconclass keywords to images...")
    keyword_images = build_keyword_to_images_map(data)
    print(f"  Found {len(keyword_images)} target keywords in the dataset.")
    keyword_counts = {k: len(v) for k, v in sorted(keyword_images.items(), key=lambda x: -len(x[1]))}
    print("  Top keywords by image count:")
    for kw, count in list(keyword_counts.items())[:20]:
        print(f"    {kw}: {count}")

    print(f"\nStep 3: Preparing YOLO classification dataset in '{DATASET_DIR}/'...")
    stats = prepare_dataset(keyword_images)
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    print(f"  Dataset ready: {len(stats)} classes, {total_train} train images, {total_val} val images.")

    print(f"\nStep 4: Fine-tuning {BASE_MODEL} on Iconclass dataset...")
    results = finetune(DATASET_DIR)
    print("\nFine-tuning complete.")
    print(f"Results saved to: {results.save_dir}")
