import hashlib
from PIL import Image

def pixelate_8x8(img: Image.Image) -> Image.Image:
    small = img.resize((8, 8), resample=Image.BILINEAR)  # type: ignore
    big = small.resize((256, 256), resample=Image.NEAREST)  # type: ignore
    return big

def url_to_filename(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"
