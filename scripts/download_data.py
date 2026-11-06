"""Download the public MSD Task01 dataset without committing patient data."""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

URL = "http://medicaldecathlon.com/files/Task01_BrainTumour.tar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "Task01_BrainTumour.tar"
    urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as handle:
        handle.extractall(args.output_dir)
    archive.unlink()


if __name__ == "__main__":
    main()
