#!/usr/bin/env python3
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path


COCO_FILES = {
    "annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
    },
    "val2017": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017.zip",
    },
    "train2017": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "filename": "train2017.zip",
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
        top_level = {name.split("/", 1)[0] for name in zf.namelist() if name and not name.startswith("__MACOSX")}
        already_extracted = all((output_dir / item).exists() for item in top_level)
        if already_extracted and not overwrite:
            print(f"Skipping already extracted archive: {zip_path}")
            return

        print(f"Extracting {zip_path} into {output_dir}")
        zf.extractall(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2017 files used by the experiments.")
    parser.add_argument("--output-dir", default="data/coco", help="Directory for COCO zip files and extracted data.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(COCO_FILES),
        default=["annotations", "val2017"],
        help="COCO files to download. Defaults to annotations and val2017.",
    )
    parser.add_argument("--include-train", action="store_true", help="Also download train2017.zip (~18 GB).")
    parser.add_argument("--no-extract", action="store_true", help="Download zip files without extracting them.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing downloads and extracted folders.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits = list(dict.fromkeys(args.splits + (["train2017"] if args.include_train else [])))

    for split in splits:
        info = COCO_FILES[split]
        zip_path = output_dir / info["filename"]
        download_file(info["url"], zip_path, overwrite=args.overwrite)
        if not args.no_extract:
            extract_zip(zip_path, output_dir, overwrite=args.overwrite)

    print(f"COCO data is available under {output_dir}")


if __name__ == "__main__":
    main()
