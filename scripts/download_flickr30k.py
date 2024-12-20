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
    res = requests.head(url, timeout=request_timeout)
    file_size_online = int(res.headers.get("content-length", -1))
    if dest_path.exists():
        file_size_offline = dest_path.stat().st_size
        if file_size_online == file_size_offline:
            log(f"Complete file {dest_path} exists. Skiping download.")
            return dest_path
        log(f"File {dest_path} is incomplete. Resuming download...")
        start_position = file_size_offline
    else:
        log(f"Downloading file to {dest_path}...")
        start_position = 0
    resume_header = (
        {"Range": f"bytes={start_position}-"} if start_position > 0 else None
    )
    res = requests.get(url, stream=True, headers=resume_header, timeout=request_timeout)
    file_mode = "ab" if start_position > 0 else "wb"
    with (
        dest_path.open(file_mode) as f,
        tqdm(
            total=None if file_size_online == -1 else file_size_online,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
            initial=start_position,
            ascii=True,
            miniters=1,
        ) as pbar,
    ):
        for chunk in res.iter_content(block_size):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest_path


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Require exactly one input arg (output dir path)"
    out_path = Path(sys.argv[1])
    out_path.mkdir(exist_ok=True, parents=True)

    log("Downloading annotations CSV")
    csv_path = download_file(url=CSV_URL, dest_dir=out_path)
    log(f"Annotations CSV is not in {csv_path}")

    log("Downloading images zip file")
    zip_path = download_file(url=ZIP_URL, dest_dir=out_path)
    log("Unzipping images")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_path)
    log("Removing zip file")
    zip_path.unlink()
