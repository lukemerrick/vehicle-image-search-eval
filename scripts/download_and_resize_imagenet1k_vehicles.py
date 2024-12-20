import logging
import sys
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from tqdm import tqdm

from src.imagenet_classifier import IMAGENET_VEHICLE_CATEGORY_IDX_ARRAY
from src.resize_img import resize_smallest_dimension

IMG_RESIZE_PX = 160

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


def resize_copy_image_subset(
    dataset: Dataset,
    indices: Sequence[int],
    out_dir: Path,
    filename_prefix: str,
    resize_px: int,
    image_key: str = "image",
) -> Sequence[Path]:
    res_paths = []
    for i in tqdm(
        indices, desc=f"exporting resized subset of {filename_prefix}", unit="image"
    ):
        img = dataset[i][image_key]
        img_smaller = resize_smallest_dimension(img, smallest_dim_px=resize_px)
        out_path = out_dir / f"{filename_prefix}_{i:08d}.jpg"
        img_smaller.save(out_path)
        res_paths.append(out_path)
    return res_paths


def get_vehicle_image_indices(
    dataset: Dataset, label_key: str = "label"
) -> Sequence[int]:
    return [
        int(i)
        for i in np.nonzero(
            pd.Series(dataset[label_key]).isin(IMAGENET_VEHICLE_CATEGORY_IDX_ARRAY)
        )[0]
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set up output directory.
    assert len(sys.argv) == 2, "Require exactly one input arg (output dir path)"
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(exist_ok=True, parents=True)

    # Download all the files.
    log("Downloading the imagenet dataset from Huggingface")
    try:
        snapshot_download(repo_id="ILSVRC/imagenet-1k", repo_type="dataset")
    except GatedRepoError:
        log(
            "You must be logged in to Huggingface and have accepted the Imagenet terms "
            "to download the data. Please accept the terms and run "
            "`huggingface-cli login` before re-running this script."
        )
        sys.exit(1)

    # Load the data.
    log("Loading imagenet from disk")
    imagenet = cast(
        DatasetDict, load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
    )

    # Filter to vehicle images and write resized images to disk.
    for subset in ("train", "validation"):
        log(f"Finding vehicle images in subset {subset}")
        dataset = imagenet[subset]
        idx = get_vehicle_image_indices(dataset)

        log(f"Writing resized images to {out_dir}")
        resize_copy_image_subset(
            dataset=dataset,
            indices=idx,
            out_dir=out_dir,
            filename_prefix=f"imagenet1k_vehicles_{subset}",
            resize_px=IMG_RESIZE_PX,
        )
