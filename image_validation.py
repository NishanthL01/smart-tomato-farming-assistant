from io import BytesIO
from typing import Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class ImageTooLargeError(ValueError):
    pass


class UploadedImage(Protocol):
    def getvalue(self) -> bytes: ...


def load_uploaded_image(uploaded_file: UploadedImage) -> Image.Image:
    image_bytes = uploaded_file.getvalue()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError("The uploaded image exceeds the 10 MB limit.")

    with Image.open(BytesIO(image_bytes)) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise UnidentifiedImageError("Only JPEG and PNG images are supported.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ImageTooLargeError("The uploaded image exceeds the 25 megapixel limit.")
        image.verify()

    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
