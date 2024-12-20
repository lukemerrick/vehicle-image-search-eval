import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

from src.imagenet_classifier import classify_vehicle, stream_images
from src.resize_img import resize_smallest_dimension

IMG_RESIZE_PX = 160
PROB_THRESHOLD = 0.6
MODEL_ID = "google/vit-base-patch16-224"

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Set up input and output filesystem structure.
    assert (
        len(sys.argv) == 3
    ), "Require exactly two input args (flicker30k dir and output dir)"
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    assert in_dir.is_dir()
    in_image_dir = in_dir / "flickr30k-images"
    in_annotations_path = in_dir / "flickr_annotations_30k.csv"
    assert in_image_dir.is_dir()
    assert in_annotations_path.is_file()
    out_dir_images = out_dir / "images"
    out_path_queries = out_dir / "queries.csv"
    out_path_qrels = out_dir / "qrels.csv"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir_images.mkdir(exist_ok=True)

    # Read the annotations.
    df = pd.read_csv(in_annotations_path)

    # Load the model.
    log(f"Loading {MODEL_ID}")
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = ViTForImageClassification.from_pretrained(MODEL_ID)
    model = model.cuda()  # type: ignore[reportCallIssue]

    # Classify the images.
    log("Classifying vehicle images from flickr30k")
    image_paths = [in_image_dir / p for p in df["filename"]]
    res = []
    for batch in stream_images(paths=image_paths, image_processor=processor):
        batch_res = classify_vehicle(model, batch.cuda())
        res.append(batch_res)
    res = np.concatenate(res)

    # Select the vehicle images.
    log(f"Using threshold {PROB_THRESHOLD} to classify vehicle vs. non-vehicle images")
    is_probably_vehicle = res > PROB_THRESHOLD
    df_vehicle = df.iloc[is_probably_vehicle]

    # Get captions.
    caption_list = [json.loads(raw) for raw in df_vehicle["raw"]]

    # Export resized vehicle images, keeping track of caption-to-image-name patterns.
    log(f"Exporting images to {out_dir}")
    qrels: list[dict[str, str | int]] = []
    queries: list[dict[str, str]] = []
    for filename, captions in zip(
        tqdm(df_vehicle["filename"]), caption_list, strict=True
    ):
        img = Image.open(in_image_dir / filename)
        img_smaller = resize_smallest_dimension(img, smallest_dim_px=IMG_RESIZE_PX)
        img_smaller.save(out_dir_images / f"flickr30k_{filename}")
        filename_no_suffix = filename.removesuffix(".jpg")
        for i, cap in enumerate(captions):
            qid = f"{filename_no_suffix}_caption_{i:02d}"
            queries.append({"query_id": qid, "query_text": cap})
            qrels.append({"query_id": qid, "document_id": filename, "relevance": 1})
    df_queries = pd.DataFrame(queries)
    df_qrels = pd.DataFrame(qrels)
    log(f"Saving {len(df_queries)} queries to {out_path_queries}")
    df_queries.to_csv(out_path_queries, index=False)
    log(f"Saving {len(df_qrels)} qrels to {out_path_qrels}")
    df_qrels.to_csv(out_path_qrels, index=False)
