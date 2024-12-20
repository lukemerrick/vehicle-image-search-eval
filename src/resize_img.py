from PIL.Image import Image
from PIL.ImageFile import ImageFile


def new_size(width: int, height: int, smallest_dim_px: int = 160) -> tuple[int, int]:
    ratio = min(1.0, smallest_dim_px / min(width, height))
    return round(ratio * width), round(ratio * height)


def resize_smallest_dimension(img: ImageFile, smallest_dim_px: int = 160) -> Image:
    return img.resize(new_size(*img.size, smallest_dim_px))
