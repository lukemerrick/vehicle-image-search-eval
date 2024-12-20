import logging
from pathlib import Path
from typing import NamedTuple, Protocol, Sequence, cast
from urllib.parse import urlparse

import torch
from PIL import Image
from torchvision.transforms import Compose

from src.download_util import download_file
from src.open_clip.factory import create_model_and_transforms
from src.open_clip.tokenizer import tokenize

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "metaclip"
MODEL_WEIGHTS_URL = (
    "https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_v1.2_altogether.pt"
)
METACLIP_1DOT2_NAME = "metaclip1.2"
METACLIP_1DOT2_BASE_ARCHITECTURE = "ViT-H-14"


logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


class TokenizeProtocol(Protocol):
    def __call__(
        self, texts: str | list[str], context_length: int = 77
    ) -> torch.LongTensor: ...


class LoadedModel(NamedTuple):
    model: torch.nn.Module
    image_processor_fn: Compose
    text_tokenizer_fn: TokenizeProtocol = tokenize


def download_meta_clip_1dot2(
    local_dir: Path = DEFAULT_CACHE_DIR, force: bool = False
) -> Path:
    filename = Path(urlparse(MODEL_WEIGHTS_URL).path).name
    local_path = local_dir / filename
    if not local_path.is_file() or force:
        log(f"{local_path} not found, downloading")
        download_res = download_file(MODEL_WEIGHTS_URL, local_dir)
        assert download_res == local_path
    return local_path


def load_model(name: str = METACLIP_1DOT2_NAME) -> LoadedModel:
    if name == METACLIP_1DOT2_NAME:
        weights_path = download_meta_clip_1dot2()
        model, _preprocess_train, preprocess = create_model_and_transforms(
            METACLIP_1DOT2_BASE_ARCHITECTURE
        )
        state_dict = torch.load(weights_path, weights_only=True)["state_dict"]
        model.load_state_dict(state_dict)
    else:
        model, _preprocess_train, preprocess = create_model_and_transforms(name)
    return LoadedModel(model=model, image_processor_fn=preprocess)


def _load_image_by_filename(preprocess_fn: Compose, path: Path | str) -> torch.Tensor:
    with Image.open(path) as img:
        image_tensor = cast(torch.Tensor, preprocess_fn(img))
        return image_tensor.unsqueeze(0)


def read_image_batch(
    image_processor_fn: Compose, image_filepaths: Sequence[Path | str]
) -> torch.Tensor:
    return torch.row_stack(
        [_load_image_by_filename(image_processor_fn, x) for x in image_filepaths]
    )
