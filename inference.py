import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from treatment_data import normalize_class_key

IMAGE_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.60


def prepare_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def run_prediction(model, labels: list[str], image: Image.Image) -> list[dict[str, object]]:
    if not labels:
        raise ValueError("The model class-name list is empty.")

    output = np.asarray(model.predict(prepare_image(image), verbose=0))
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("The model must return one two-dimensional prediction row.")
    if output.shape[1] != len(labels):
        raise ValueError(
            "The number of model outputs does not match the class names. "
            "Check tomato_class_names.json."
        )
    predictions = output[0]
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned a non-finite prediction score.")

    top_indices = np.argsort(predictions)[::-1][:3]
    return [
        {
            "raw_key": str(labels[int(index)]),
            "key": normalize_class_key(labels[int(index)]),
            "score": float(predictions[int(index)]),
        }
        for index in top_indices
    ]


def is_confident_prediction(score: float) -> bool:
    return score >= CONFIDENCE_THRESHOLD
