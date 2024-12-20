import logging
import sys
import zipfile
from pathlib import Path

from src.download_util import download_file

ZIP_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
CSV_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv"

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


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
