#!/usr/bin/env python3
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path


VQAV2_FILES = {
    "val_images": {
        "url": "http://images.cocodataset.org/zips/val2014.zip",
        "filename": "val2014.zip",
    },
    "val_questions": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "filename": "v2_Questions_Val_mscoco.zip",
    },
    "val_annotations": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "filename": "v2_Annotations_Val_mscoco.zip",
    },
    "train_images": {
        "url": "http://images.cocodataset.org/zips/train2014.zip",
        "filename": "train2014.zip",
    },
    "train_questions": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "filename": "v2_Questions_Train_mscoco.zip",
    },
    "train_annotations": {
        "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "filename": "v2_Annotations_Train_mscoco.zip",
    },
}


def download_file(url, output_path, overwrite=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return

    print(f"Downloading {url}")
    print(f"  -> {output_path}")

    def report_progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size <= 0:
            sys.stdout.write(f"\r  downloaded {downloaded / (1024 ** 2):.1f} MB")
        else:
            pct = min(downloaded / total_size, 1.0) * 100.0
            sys.stdout.write(
                f"\r  {pct:5.1f}% ({downloaded / (1024 ** 2):.1f} / {total_size / (1024 ** 2):.1f} MB)"
            )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    sys.stdout.write("\n")


def extract_zip(zip_path, output_dir, overwrite=False):
    if not zip_path.exists():
        raise FileNotFoundError(f"Cannot extract missing file: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        top_level = {
            name.split("/", 1)[0]
            for name in zf.namelist()
            if name and not name.startswith("__MACOSX")
        }
        already_extracted = all((output_dir / item).exists() for item in top_level)
        if already_extracted and not overwrite:
            print(f"Skipping already extracted archive: {zip_path}")
            return

        print(f"Extracting {zip_path} into {output_dir}")
        zf.extractall(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download VQAv2 files used by the ViLT reproduction.")
    parser.add_argument("--output-dir", default="data/vqav2", help="Directory for VQAv2 zip files and extracted data.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(VQAV2_FILES),
        default=["val_images", "val_questions", "val_annotations"],
        help="Files to download. Defaults to VQAv2 val images, questions, and annotations.",
    )
    parser.add_argument("--include-train", action="store_true", help="Also download VQAv2 train2014 files.")
    parser.add_argument("--no-extract", action="store_true", help="Download zip files without extracting them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extracted files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits = list(args.splits)
    if args.include_train:
        splits.extend(["train_images", "train_questions", "train_annotations"])
    splits = list(dict.fromkeys(splits))

    for split in splits:
        info = VQAV2_FILES[split]
        zip_path = output_dir / info["filename"]
        download_file(info["url"], zip_path, overwrite=args.overwrite)
        if not args.no_extract:
            extract_zip(zip_path, output_dir, overwrite=args.overwrite)

    print(f"VQAv2 data is available under {output_dir}")


if __name__ == "__main__":
    main()
