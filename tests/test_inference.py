import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from inference import is_confident_prediction, prepare_image, run_prediction


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 24), color=(0, 128, 255))
        self.labels = [f"Tomato___class_{index}" for index in range(10)]

    def test_preprocessing_preserves_the_existing_mobilenet_pipeline(self):
        prepared = prepare_image(self.image)

        self.assertEqual(prepared.shape, (1, 160, 160, 3))
        expected = np.array((-1.0, 128.0 / 127.5 - 1.0, 1.0), dtype=np.float32)
        np.testing.assert_allclose(prepared[0, 0, 0], expected, atol=1e-6)

    def test_output_indices_stay_aligned_with_class_name_order(self):
        scores = np.linspace(0.01, 0.10, 10, dtype=np.float32)
        model = Mock()
        model.predict.return_value = scores.reshape(1, -1)

        results = run_prediction(model, self.labels, self.image)

        self.assertEqual([item["raw_key"] for item in results], self.labels[9:6:-1])
        self.assertEqual([item["key"] for item in results], ["class_9", "class_8", "class_7"])
        self.assertEqual([item["score"] for item in results], [float(value) for value in scores[9:6:-1]])

    def test_output_count_and_shape_are_validated(self):
        model = Mock()
        model.predict.return_value = np.zeros((1, 9), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_prediction(model, self.labels, self.image)

        model.predict.return_value = np.zeros((1, 1, 10), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            run_prediction(model, self.labels, self.image)

    def test_nonfinite_scores_and_empty_labels_are_rejected(self):
        model = Mock()
        model.predict.return_value = np.array([[float("nan")] + [0.1] * 9])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            run_prediction(model, self.labels, self.image)
        with self.assertRaisesRegex(ValueError, "empty"):
            run_prediction(model, [], self.image)

    def test_low_confidence_is_not_treated_as_a_confirmed_prediction(self):
        self.assertFalse(is_confident_prediction(0.59))
        self.assertTrue(is_confident_prediction(0.60))


if __name__ == "__main__":
    unittest.main()
