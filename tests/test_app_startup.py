import importlib
import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

import farm_data


PROJECT_DIR = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_farmer_navigation_and_task_completion(self):
        previous_data_dir = os.environ.get("TOMATO_APP_DATA_DIR")
        with tempfile.TemporaryDirectory() as data_dir:
            os.environ["TOMATO_APP_DATA_DIR"] = data_dir
            importlib.reload(farm_data)
            try:
                app = AppTest.from_file(str(PROJECT_DIR / "app.py")).run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                self.assertTrue(any(item.value == "Welcome to FarmAI" for item in app.title))
                self.assertTrue(any(button.label == "Check My Plant" for button in app.button))
                self.assertTrue(any(button.label == "Water My Crop" for button in app.button))

                next(button for button in app.button if button.label == "Water My Crop").click().run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Irrigation Advisor" in item.value for item in app.title))

                app.radio[0].set_value("My Farm Tasks").run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Today" in item.value for item in app.markdown))
                self.assertTrue(any("Coming up" in item.value for item in app.markdown))
                self.assertTrue(any("Completed" in item.value for item in app.markdown))

                task_title = next(widget for widget in app.text_input if widget.label == "Task title")
                task_title.set_value("Check tomato plants")
                save_button = next(button for button in app.button if button.label == "Save task")
                save_button.click().run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Check tomato plants" in item.value for item in app.markdown))

                done_button = next(button for button in app.button if button.label == "Mark as done")
                done_button.click().run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(farm_data.list_tasks()[0]["status"], "completed")

                app.radio[0].set_value("Check Plant").run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("Step 1 — Take a photo of your plant" in item.value for item in app.markdown))
                self.assertTrue(any("An AI result is not a guaranteed diagnosis." in item.value for item in app.markdown))

                next(widget for widget in app.selectbox if widget.label == "தமிழ் | English").set_value("தமிழ்").run(timeout=120)
                self.assertTrue(any("படி 1" in item.value for item in app.markdown))
                next(button for button in app.button if button.label == "🏠 முகப்பு").click().run(timeout=120)
                self.assertTrue(any(item.value == "FarmAI-க்கு வரவேற்கிறோம்" for item in app.title))
                next(button for button in app.button if button.label == "இன்றைய வானிலை").click().run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("வானிலை முன்னறிவிப்பு" in item.value for item in app.title))

                app.radio[0].set_value("மேலும்").run(timeout=120)
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(button.label == "நீர்ப்பாசன ஆலோசனை" for button in app.button))
            finally:
                if previous_data_dir is None:
                    os.environ.pop("TOMATO_APP_DATA_DIR", None)
                else:
                    os.environ["TOMATO_APP_DATA_DIR"] = previous_data_dir
                importlib.reload(farm_data)


if __name__ == "__main__":
    unittest.main()
