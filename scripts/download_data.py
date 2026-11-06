"""Download and validate the public MSD Task01 dataset."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
KAGGLE_DATASET = "awsaf49/brats2020-training-data"
EXPECTED_DIRECTORIES = ("imagesTr", "labelsTr")


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "brain-mri-fem/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _validate_dataset(root: Path) -> None:
    missing = [directory for directory in EXPECTED_DIRECTORIES if not (root / directory).is_dir()]
    if missing:
        raise RuntimeError(
            f"Dataset extraction is incomplete; missing directories: {', '.join(missing)}"
        )
    images = list((root / "imagesTr").glob("*.nii.gz"))
    labels = list((root / "labelsTr").glob("*.nii.gz"))
    if not images or not labels:
        raise RuntimeError("The dataset contains no training NIfTI images or labels")
    print(f"Validated {len(images)} training images and {len(labels)} labels in {root}")


def _download_from_kaggle(dataset: str, output_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is required for Kaggle downloads. Install it with "
            "'python -m pip install kagglehub'."
        ) from error
    downloaded_path = Path(kagglehub.dataset_download(dataset))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / dataset.rsplit("/", maxsplit=1)[-1]
    if downloaded_path.resolve() != destination.resolve():
        shutil.copytree(downloaded_path, destination, dirs_exist_ok=True)
    nifti_files = list(destination.rglob("*.nii.gz"))
    h5_files = list(destination.rglob("*.h5"))
    if not nifti_files and not h5_files:
        raise RuntimeError(f"No .nii.gz or .h5 files found in downloaded dataset: {destination}")
    if h5_files and not nifti_files:
        print(
            f"Copied {len(h5_files)} HDF5 2D slices to: {destination}. "
            "This Kaggle version is for 2D training; use raw NIfTI BraTS data for 3D reconstruction."
        )
    else:
        print(f"Copied {len(nifti_files)} NIfTI files to: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a brain MRI dataset")
    parser.add_argument("--source", choices=("msd", "kaggle"), default="kaggle")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=URL, help="Dataset archive URL")
    parser.add_argument("--dataset", default=KAGGLE_DATASET, help="Kaggle dataset identifier")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    if args.source == "kaggle":
        _download_from_kaggle(args.dataset, args.output_dir)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "Task01_BrainTumour.tar"
    print(f"Downloading {args.url}")
    _download(args.url, archive)
    with tarfile.open(archive) as handle:
        handle.extractall(args.output_dir, filter="data")
    dataset_root = args.output_dir / "Task01_BrainTumour"
    _validate_dataset(dataset_root)
    if not args.keep_archive:
        archive.unlink()


if __name__ == "__main__":
    main()
