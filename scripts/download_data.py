"""Download NepCOV19Tweets from the dataset authors' Kaggle record and verify it."""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/mathew11111/nepcov19tweets"
SHA256 = "eba73b8ca8bf359d1b39bdfa3e9d9a7d72f8e21602f2c762163d3234ba484851"
OUT = Path("data/raw/nepcov19tweets")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "archive.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: {digest}; the upstream release may have changed")
    with zipfile.ZipFile(archive) as z:
        z.extractall(OUT)
    print(f"extracted to {OUT}")


if __name__ == "__main__":
    main()
