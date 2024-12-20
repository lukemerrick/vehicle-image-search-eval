import logging
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

ZIP_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
CSV_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv"

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


def download_file(
    url: str, dest_dir: Path, block_size: int = 32 * 1024, request_timeout: int = 10
) -> Path:
    dest_dir.mkdir(exist_ok=True, parents=True)
    dest_path = dest_dir / Path(urlparse(url).path).name
    log(f"Downloading file to {dest_path}...")
    res = requests.get(url, stream=True, timeout=request_timeout)
    with (
        dest_path.open("wb") as f,
        tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name) as pbar,
    ):
        for chunk in res.iter_content(block_size):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set up output directory.
    assert len(sys.argv) == 2, "Require exactly one input arg (output dir path)"
    out_path = Path(sys.argv[1])
    out_path.mkdir(exist_ok=True, parents=True)

    # Downloads.
    log("Downloading annotations CSV")
    csv_path = download_file(url=CSV_URL, dest_dir=out_path)
    log(f"Annotations CSV is now in {csv_path}")
    log("Downloading images zip file")
    zip_path = download_file(url=ZIP_URL, dest_dir=out_path)

    # Unzip images.
    log("Unzipping images")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)
    log("Removing zip file")
    zip_path.unlink()
