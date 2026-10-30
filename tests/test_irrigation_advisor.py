import unittest
from datetime import date

from irrigation_advisor import build_irrigation_advice


class IrrigationAdvisorTests(unittest.TestCase):
    def test_advice_discloses_inputs_uncertainty_and_no_water_quantity(self):
        advice = build_irrigation_advice(
            "Tomato",
            "Flowering",
            "Loam",
            date(2026, 2, 1),
            forecast_rain_probability=60,
        )
        self.assertEqual(advice["inputs"]["last_irrigation_date"], "2026-02-01")
        self.assertEqual(advice["inputs"]["forecast_rain_probability"], 60)
        self.assertTrue(advice["uncertainty"])
        self.assertTrue(any("forecast only" in reason for reason in advice["reasoning"]))
        self.assertNotIn("litre", advice["action"].casefold())
        self.assertNotIn("gallon", advice["action"].casefold())

    def test_missing_field_details_are_explained_without_claiming_sensor_data(self):
        advice = build_irrigation_advice("Tomato", None, None, None)
        self.assertIn("crop stage", advice["reasoning"][-1])
        self.assertIn("soil type", advice["reasoning"][-1])
        self.assertIn("last irrigation date", advice["reasoning"][-1])
        self.assertIn("no soil-moisture sensor", advice["reasoning"][1])


if __name__ == "__main__":
    unittest.main()
