import json
import unittest
from pathlib import Path

from treatment_data import find_treatment_entry, get_treatment_options


PROJECT_DIR = Path(__file__).resolve().parents[1]


class TreatmentDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (PROJECT_DIR / "pesticide_database.json").open(encoding="utf-8") as file:
            cls.database = json.load(file)
        with (PROJECT_DIR / "tomato_class_names.json").open(encoding="utf-8") as file:
            cls.labels = json.load(file)

    def test_every_model_label_has_a_matching_knowledge_entry(self):
        for label in self.labels:
            with self.subTest(label=label):
                self.assertIsNotNone(find_treatment_entry(self.database, label))

    def test_database_schema_returns_natural_and_chemical_options(self):
        entry = find_treatment_entry(self.database, "Tomato___Early_blight")

        self.assertIsNotNone(entry)
        self.assertEqual(len(get_treatment_options(entry, "natural_options")), 2)
        self.assertEqual(len(get_treatment_options(entry, "chemical_options")), 2)

    def test_unverified_chemical_dose_stays_unverified(self):
        entry = find_treatment_entry(self.database, "Tomato___Early_blight")
        chemical_options = get_treatment_options(entry, "chemical_options")

        self.assertTrue(all(option.get("verified") is False for option in chemical_options))
        self.assertTrue(all(option.get("dose") is None for option in chemical_options))

    def test_unknown_and_malformed_entries_are_ignored(self):
        self.assertIsNone(find_treatment_entry({}, "Tomato___unknown"))
        self.assertEqual(get_treatment_options({"natural_options": "invalid"}, "natural_options"), [])


if __name__ == "__main__":
    unittest.main()
