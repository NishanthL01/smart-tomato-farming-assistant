import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image, UnidentifiedImageError

from image_validation import (
    MAX_IMAGE_BYTES,
    ImageTooLargeError,
    load_uploaded_image,
)


class ImageValidationTests(unittest.TestCase):
    def make_png(self):
        image_bytes = BytesIO()
        Image.new("RGB", (2, 3), color="green").save(image_bytes, format="PNG")
        return image_bytes.getvalue()

    def test_valid_png_is_loaded_as_rgb(self):
        image = load_uploaded_image(BytesIO(self.make_png()))

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 3))

    def test_invalid_image_bytes_are_rejected(self):
        with self.assertRaises(UnidentifiedImageError):
            load_uploaded_image(BytesIO(b"not an image"))

    def test_byte_limit_is_enforced_before_decode(self):
        uploaded = BytesIO(b"x" * (MAX_IMAGE_BYTES + 1))

        with self.assertRaises(ImageTooLargeError):
            load_uploaded_image(uploaded)

    @patch("image_validation.Image.open")
    def test_pixel_limit_is_enforced_before_decode(self, image_open):
        image = MagicMock()
        image.format = "JPEG"
        image.width = 5_001
        image.height = 5_000
        context_manager = MagicMock()
        context_manager.__enter__.return_value = image
        image_open.return_value = context_manager

        with self.assertRaises(ImageTooLargeError):
            load_uploaded_image(BytesIO(b"image data"))

        image.verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
