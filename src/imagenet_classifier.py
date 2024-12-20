from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

IMAGENET_VEHICLE_CATEGORIES = {
    "Model T": 661,
    "ambulance": 407,
    "amphibian, amphibious vehicle": 408,
    "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon": 436,  # noqa: E501
    "bicycle-built-for-two, tandem bicycle, tandem": 444,
    "bobsled, bobsleigh, bob": 450,
    "cab, hack, taxi, taxicab": 468,
    "convertible": 511,
    "crane": 517,
    "dogsled, dog sled, dog sleigh": 537,
    "fire engine, fire truck": 555,
    "forklift": 561,
    "garbage truck, dustcart": 569,
    "go-kart": 573,
    "golfcart, golf cart": 575,
    "grille, radiator grille": 581,
    "half track": 586,
    "harvester, reaper": 595,
    "horse cart, horse-cart": 603,
    "jeep, landrover": 609,
    "jinrikisha, ricksha, rickshaw": 612,
    "lawn mower, mower": 621,
    "limousine, limo": 627,
    "minibus": 654,
    "minivan": 656,
    "moped": 665,
    "motor scooter, scooter": 670,
    "mountain bike, all-terrain bike, off-roader": 671,
    "moving van": 675,
    "oxcart": 690,
    "passenger car, coach, carriage": 705,
    "pickup, pickup truck": 717,
    "plow, plough": 730,
    "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria": 734,
    "racer, race car, racing car": 751,
    "recreational vehicle, RV, R.V.": 757,
    "school bus": 779,
    "snowmobile": 802,
    "snowplow, snowplough": 803,
    "sports car, sport car": 817,
    "streetcar, tram, tramcar, trolley, trolley car": 829,
    "tank, army tank, armored combat vehicle, armoured combat vehicle": 847,
    "thresher, thrasher, threshing machine": 856,
    "tow truck, tow car, wrecker": 864,
    "tractor": 866,
    "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi": 867,
    "tricycle, trike, velocipede": 870,
    "trolleybus, trolley coach, trackless trolley": 874,
    "unicycle, monocycle": 880,
}
IMAGENET_VEHICLE_CATEGORY_IDX_ARRAY = np.asarray(
    list(IMAGENET_VEHICLE_CATEGORIES.values())
)


def stream_images(
    paths: Sequence[Path],
    image_processor: ViTImageProcessor,
    batch_size: int = 256,
    progress_bar: bool = True,
) -> Iterable[Tensor]:
    batched_paths = [
        paths[i : i + batch_size] for i in range(0, len(paths), batch_size)
    ]
    with tqdm(total=len(paths), unit="image", disable=not progress_bar) as pbar:
        for path_batch in batched_paths:
            pixel_values = []
            for p in path_batch:
                with Image.open(p) as img:
                    img_dict = image_processor(img, return_tensors="pt")
                pixel_values.append(img_dict["pixel_values"][0].unsqueeze(0))
            result = torch.cat(pixel_values, dim=0)
            pbar.update(result.size(0))
            yield result


@torch.inference_mode()
def classify_images(
    model: ViTForImageClassification, batch: Tensor
) -> NDArray[np.int64]:
    return model(pixel_values=batch).logits.argmax(-1).cpu().numpy()


@torch.inference_mode()
def classify_vehicle(
    model: ViTForImageClassification, batch: Tensor
) -> NDArray[np.float32]:
    probs = torch.softmax(model(pixel_values=batch).logits, dim=-1).cpu().numpy()
    return probs[:, IMAGENET_VEHICLE_CATEGORY_IDX_ARRAY].sum(axis=1)
