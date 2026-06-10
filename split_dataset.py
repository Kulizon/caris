import json
import os
import random
import shutil
import zipfile

DATASET_ZIP = os.path.join("datasets", "iconclass_dataset.zip")
TUNE_DIR = "tune_data"
EVAL_DIR = "eval_data"
SEED = 37
EVAL_FRACTION = 0.0115


def split_dataset(
    zip_path: str = DATASET_ZIP,
    tune_dir: str = TUNE_DIR,
    eval_dir: str = EVAL_DIR,
    eval_fraction: float = EVAL_FRACTION,
    seed: int = SEED,
):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")

    for d in (tune_dir, eval_dir):
        os.makedirs(os.path.join(d, "images"), exist_ok=True)

    print(f"Reading data.json from {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("data.json") as f:
            full_data: dict = json.load(f)

    image_keys = list(full_data.keys())
    print(f"Total images in dataset: {len(image_keys)}")

    random.seed(seed)
    random.shuffle(image_keys)

    n_eval = max(1, int(len(image_keys) * eval_fraction))
    print(f"  Total images: {len(image_keys)}")
    eval_keys = set(image_keys[:n_eval])
    tune_keys = set(image_keys[n_eval:])

    print(f"Eval split: {len(eval_keys)} images")
    print(f"Tune split: {len(tune_keys)} images")

    eval_data = {k: full_data[k] for k in eval_keys}
    tune_data = {k: full_data[k] for k in tune_keys}

    eval_json_path = os.path.join(eval_dir, "data.json")
    tune_json_path = os.path.join(tune_dir, "data.json")

    with open(eval_json_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Wrote {eval_json_path}")

    with open(tune_json_path, "w") as f:
        json.dump(tune_data, f, indent=2)
    print(f"Wrote {tune_json_path}")

    print("Extracting images (this may take a while) ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members):
            if not member.endswith(".jpg"):
                continue

            basename = os.path.basename(member)
            if basename in eval_keys:
                dest = os.path.join(eval_dir, "images", basename)
            elif basename in tune_keys:
                dest = os.path.join(tune_dir, "images", basename)
            else:
                continue

            if not os.path.exists(dest):
                with zf.open(member) as src_f, open(dest, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)

            if (i + 1) % 10000 == 0:
                print(f"  ... extracted {i + 1}/{len(members)} entries")

    n_eval_imgs = len(os.listdir(os.path.join(eval_dir, "images")))
    n_tune_imgs = len(os.listdir(os.path.join(tune_dir, "images")))
    print(f"Done. eval_data/images: {n_eval_imgs}, tune_data/images: {n_tune_imgs}")


if __name__ == "__main__":
    split_dataset()
