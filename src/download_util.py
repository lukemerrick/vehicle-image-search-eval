import logging
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


def download_file(
    url: str, dest_dir: Path, block_size: int = 32 * 1024, request_timeout: int = 10
) -> Path:
    """Download a file from the given URL to the given local directory.

    Returns the local path to the downloaded file.
    """
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
