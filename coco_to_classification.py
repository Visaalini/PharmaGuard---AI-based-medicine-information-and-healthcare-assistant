"""
coco_to_classification.py
─────────────────────────────────────────────────────────────────────────────
Converts a COCO-format medicine strip dataset into a flat classification
folder structure:

  <split>/real/  ← images whose majority category is 'authentic'  (category_id=1)
  <split>/fake/  ← images whose majority category is 'counterfeit' (category_id=2)

Supports: train / valid / test splits (auto-detected from SOURCE_BASE).

Usage:
  python coco_to_classification.py
  python coco_to_classification.py --src dataset --dst dataset_cls --splits train valid test
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


# ── Category mapping (from _annotations.coco.json) ───────────────────────────
# id=1  → 'authentic'   → real
# id=2  → 'counterfeit' → fake
AUTHENTIC_NAMES  = {"authentic"}
COUNTERFEIT_NAMES = {"counterfeit", "counterfeit-medicine"}


def load_coco(annotation_path: Path) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_category_map(coco: dict) -> dict:
    """
    Returns {category_id: 'real'|'fake'|None} based on category name.
    """
    mapping = {}
    for cat in coco.get("categories", []):
        name = cat["name"].lower().strip()
        cat_id = cat["id"]
        if name in AUTHENTIC_NAMES:
            mapping[cat_id] = "real"
        elif name in COUNTERFEIT_NAMES:
            mapping[cat_id] = "fake"
        else:
            mapping[cat_id] = None  # supercategory / unknown → skip
    return mapping


def build_image_label_map(coco: dict, cat_map: dict) -> dict:
    """
    For each image, collect all annotation category labels and
    decide the majority class (real/fake).  Ties go to 'fake' (conservative).

    Returns {image_id: 'real'|'fake'}
    """
    image_votes: dict[int, Counter] = {}
    for ann in coco.get("annotations", []):
        img_id  = ann["image_id"]
        label   = cat_map.get(ann["category_id"])
        if label is None:
            continue
        image_votes.setdefault(img_id, Counter())
        image_votes[img_id][label] += 1

    result = {}
    for img_id, votes in image_votes.items():
        if not votes:
            continue
        # majority vote; tie → fake (safer for counterfeit detection)
        real_v = votes.get("real", 0)
        fake_v = votes.get("fake", 0)
        result[img_id] = "real" if real_v > fake_v else "fake"
    return result


def convert_split(src_dir: Path, dst_dir: Path, split: str) -> None:
    """
    Process one split folder (e.g. dataset/train → dataset_cls/train).
    """
    split_src = src_dir / split
    annotation_file = split_src / "_annotations.coco.json"

    if not annotation_file.exists():
        print(f"  [SKIP] {annotation_file} not found.")
        return

    print(f"\n-- Processing split: {split} --------------------------")
    coco    = load_coco(annotation_file)
    cat_map = build_category_map(coco)
    print(f"  Category map: {cat_map}")

    img_label_map = build_image_label_map(coco, cat_map)

    # Build image_id → filename lookup
    id_to_file: dict[int, str] = {img["id"]: img["file_name"] for img in coco["images"]}

    # Create destination sub-folders
    real_dst = dst_dir / split / "real"
    fake_dst = dst_dir / split / "fake"
    real_dst.mkdir(parents=True, exist_ok=True)
    fake_dst.mkdir(parents=True, exist_ok=True)

    copied_real = 0
    copied_fake = 0
    skipped     = 0

    for i, (img_id, label) in enumerate(img_label_map.items()):
        filename = id_to_file.get(img_id)
        if not filename:
            skipped += 1
            continue

        src_img = split_src / filename
        if not src_img.exists():
            print(f"  [WARN] Image not found: {src_img}")
            skipped += 1
            continue

        if label == "real":
            dst_path = real_dst / filename
            copied_real += 1
        else:
            dst_path = fake_dst / filename
            copied_fake += 1

        shutil.copy2(src_img, dst_path)
        
        # Print progress every 500 images
        if i > 0 and i % 500 == 0:
            print(f"  ... copied {i} images so far ...")

    total_annotated = len(img_label_map)
    total_images    = len(coco["images"])

    print(f"  Images in JSON  : {total_images}")
    print(f"  Annotated images: {total_annotated}")
    print(f"  -> real/        : {copied_real}")
    print(f"  -> fake/        : {copied_fake}")
    print(f"  Skipped (miss)  : {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO medicine dataset to classification folders."
    )
    parser.add_argument(
        "--src",
        default="dataset",
        help="Source dataset root directory (default: dataset/)",
    )
    parser.add_argument(
        "--dst",
        default="dataset_cls",
        help="Destination classification root (default: dataset_cls/)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Which splits to process (default: train valid test)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    src_dir  = Path(args.src) if Path(args.src).is_absolute() else base_dir / args.src
    dst_dir  = Path(args.dst) if Path(args.dst).is_absolute() else base_dir / args.dst

    print("=" * 60)
    print("PharmaGuard - COCO to Classification Converter")
    print("=" * 60)
    print(f"Source : {src_dir}")
    print(f"Dest   : {dst_dir}")
    print(f"Splits : {args.splits}")

    for split in args.splits:
        convert_split(src_dir, dst_dir, split)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output written to: {dst_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
