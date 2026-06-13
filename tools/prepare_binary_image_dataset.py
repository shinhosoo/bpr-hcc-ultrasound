#!/usr/bin/env python3
import argparse
import os
import pickle
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_TO_LABEL = {
    "Benign": 0,
    "Malignant": 1,
}


def collect_images(source_dir):
    samples = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name, label))
    if not samples:
        raise RuntimeError(f"No images found under {source_dir}")
    return samples


def split_by_class(samples, val_ratio, seed):
    rng = random.Random(seed)
    train, val = [], []
    for class_name in CLASS_TO_LABEL:
        class_samples = [sample for sample in samples if sample[1] == class_name]
        rng.shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * val_ratio))
        val.extend(class_samples[:val_count])
        train.extend(class_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_imagefolder(samples, split_name, out_dir):
    split_dir = out_dir / "imagefolder" / split_name
    for class_name in CLASS_TO_LABEL:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    copied = []
    for src, class_name, label in samples:
        dst = split_dir / class_name / src.name
        shutil.copy2(src, dst)
        copied.append({"img_root": str(dst.resolve()), "label": label})
    return copied


def write_pickle(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(records, f)


def write_summary(train, val, out_dir):
    lines = [
        "Binary lesion dataset prepared from image/",
        "",
        "Labels:",
        "  Benign: 0",
        "  Malignant: 1",
        "",
        f"Train: {len(train)}",
        f"Val: {len(val)}",
    ]
    for split_name, split in [("Train", train), ("Val", val)]:
        counts = {class_name: 0 for class_name in CLASS_TO_LABEL}
        for _, class_name, _ in split:
            counts[class_name] += 1
        lines.append(f"{split_name} counts: {counts}")
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare Benign/Malignant PNG folders for 2D binary training.")
    parser.add_argument("--source", default="image", help="Folder containing Benign and Malignant subfolders.")
    parser.add_argument("--out", default="prepared_binary_lesion", help="Output dataset folder.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio per class.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    samples = collect_images(source_dir)
    train, val = split_by_class(samples, args.val_ratio, args.seed)

    reset_dir(out_dir)
    train_records = copy_imagefolder(train, "train", out_dir)
    val_records = copy_imagefolder(val, "val", out_dir)
    write_pickle(train_records, out_dir / "pkl" / "lesion_train.pkl")
    write_pickle(val_records, out_dir / "pkl" / "lesion_val.pkl")
    write_summary(train, val, out_dir)

    print(f"Prepared {len(train)} train and {len(val)} validation images in {out_dir}")
    print(f"MedViT train folder: {out_dir / 'imagefolder' / 'train'}")
    print(f"MedViT val folder: {out_dir / 'imagefolder' / 'val'}")
    print(f"DiffMIC train pkl: {out_dir / 'pkl' / 'lesion_train.pkl'}")
    print(f"DiffMIC val pkl: {out_dir / 'pkl' / 'lesion_val.pkl'}")


if __name__ == "__main__":
    main()
