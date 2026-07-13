"""Fetch COCO images for colorization (annotations are not needed).

Two sources:

  1. `zip` (default, works anywhere): downloads the COCO 2017 *val* split
     (~5k images, ~780 MB) and extracts it into --data-root. Small enough for
     Colab/Kaggle and enough for a proof-of-concept train/val split.

        python download_data.py --data-root data/coco --max-samples 2000

  2. `fiftyone` (optional, for a bigger train subset): pulls N images from the
     COCO 2017 *train* split via the FiftyOne zoo. Requires `pip install fiftyone`.

        python download_data.py --source fiftyone --max-samples 20000 --data-root data/coco

The colorization dataset just globs images under --data-root, so any folder of
RGB photos works too.
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, count * block_size * 100 // total_size)
    print(f"\r  downloading... {pct}%", end="", flush=True)


def download_zip(data_root: str, max_samples: int):
    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "val2017.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {VAL2017_URL}")
        urllib.request.urlretrieve(VAL2017_URL, zip_path, _progress)
        print()
    else:
        print(f"Using cached {zip_path}")

    extract_dir = os.path.join(data_root, "images")
    os.makedirs(extract_dir, exist_ok=True)
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".jpg")]
        if max_samples > 0:
            members = members[:max_samples]
        for m in members:
            target = os.path.join(extract_dir, os.path.basename(m))
            if os.path.exists(target):
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    n = len([f for f in os.listdir(extract_dir) if f.lower().endswith(".jpg")])
    print(f"Ready: {n} images in {extract_dir}")


def download_fiftyone(data_root: str, max_samples: int):
    try:
        import fiftyone.zoo as foz
    except ImportError:
        raise SystemExit("fiftyone not installed. Run: pip install fiftyone")
    print(f"Loading COCO-2017 train subset ({max_samples} samples) via FiftyOne...")
    ds = foz.load_zoo_dataset(
        "coco-2017", split="train",
        max_samples=max_samples if max_samples > 0 else None,
        shuffle=True, label_types=[],
    )
    extract_dir = os.path.join(data_root, "images")
    os.makedirs(extract_dir, exist_ok=True)
    for sample in ds:
        dst = os.path.join(extract_dir, os.path.basename(sample.filepath))
        if not os.path.exists(dst):
            shutil.copy(sample.filepath, dst)
    n = len(os.listdir(extract_dir))
    print(f"Ready: {n} images in {extract_dir}")


def main():
    ap = argparse.ArgumentParser(description="Download COCO images for colorization")
    ap.add_argument("--data-root", default="data/coco")
    ap.add_argument("--source", choices=["zip", "fiftyone"], default="zip")
    ap.add_argument("--max-samples", type=int, default=0,
                    help=">0 limits the number of images (0 = all)")
    args = ap.parse_args()
    if args.source == "zip":
        download_zip(args.data_root, args.max_samples)
    else:
        download_fiftyone(args.data_root, args.max_samples)


if __name__ == "__main__":
    main()
