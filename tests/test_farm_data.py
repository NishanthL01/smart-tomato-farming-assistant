import tempfile
import unittest
from datetime import date
from pathlib import Path
import sqlite3

from farm_data import (
    calculate_financial_summary,
    confirm_scan,
    delete_local_records,
    delete_task,
    delete_plot,
    delete_scan,
    initialize_database,
    latest_irrigation_date,
    list_financial_records,
    list_plots,
    list_scans,
    list_soil_tests,
    list_tasks,
    save_plot,
    save_financial_record,
    save_irrigation_log,
    save_scan,
    save_soil_test,
    save_task,
    set_task_status,
)


class FarmDataTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "farm.sqlite3"
        self.plot = {
            "farm_name": "North Farm",
            "plot_name": "Tomato A",
            "village": "Test Village",
            "district": "Test District",
            "state": "Tamil Nadu",
            "tomato_variety": "Variety 1",
            "planting_date": date(2026, 1, 15),
            "area": 1.5,
            "area_unit": "acres",
            "soil_type": "Loam",
            "cultivation_type": "Open field",
            "irrigation_method": "Drip",
            "crop_stage": "Vegetative",
            "notes": "Test notes",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_plot_persists_and_can_be_updated(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        loaded = list_plots(self.database_path)
        self.assertEqual(loaded[0]["id"], plot_id)
        self.assertEqual(loaded[0]["plot_name"], "Tomato A")
        self.assertEqual(loaded[0]["planting_date"], "2026-01-15")

        updated_plot = {**self.plot, "plot_name": "Tomato B"}
        save_plot(updated_plot, plot_id=plot_id, database_path=self.database_path)
        self.assertEqual(list_plots(self.database_path)[0]["plot_name"], "Tomato B")

    def test_invalid_required_fields_and_area_are_rejected(self):
        with self.assertRaises(ValueError):
            save_plot({**self.plot, "farm_name": " "}, database_path=self.database_path)
        with self.assertRaises(ValueError):
            save_plot({**self.plot, "area": float("inf")}, database_path=self.database_path)

    def test_plot_can_be_deleted(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        delete_plot(plot_id, self.database_path)
        self.assertEqual(list_plots(self.database_path), [])

    def test_scan_prediction_and_farmer_confirmation_remain_distinct(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        results = [
            {"key": "Early_blight", "score": 0.8},
            {"key": "Late_blight", "score": 0.1},
            {"key": "healthy", "score": 0.05},
        ]
        scan_id = save_scan(plot_id, results, self.database_path)

        scan = list_scans(self.database_path)[0]
        self.assertEqual(scan["plot_id"], plot_id)
        self.assertEqual(scan["predicted_label"], "Early_blight")
        self.assertIsNone(scan["farmer_confirmed_label"])
        self.assertEqual(scan["alternatives"][0]["label"], "Late_blight")

        confirm_scan(scan_id, "Late_blight", self.database_path)
        confirmed_scan = list_scans(self.database_path)[0]
        self.assertEqual(confirmed_scan["predicted_label"], "Early_blight")
        self.assertEqual(confirmed_scan["farmer_confirmed_label"], "Late_blight")

        delete_plot(plot_id, self.database_path)
        unlinked_scan = list_scans(self.database_path)[0]
        self.assertIsNone(unlinked_scan["plot_id"])
        self.assertIsNone(unlinked_scan["farm_name"])
        delete_scan(scan_id, self.database_path)
        self.assertEqual(list_scans(self.database_path), [])

    def test_new_schema_upgrade_preserves_existing_plot_and_scan_rows(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        scan_id = save_scan(
            plot_id, [{"key": "Early_blight", "score": 0.8}], self.database_path
        )
        initialize_database(self.database_path)
        self.assertEqual(list_plots(self.database_path)[0]["id"], plot_id)
        self.assertEqual(list_scans(self.database_path)[0]["id"], scan_id)

    def test_task_edit_deduplication_status_and_delete(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        task_id = save_task(plot_id, "Check soil", "irrigation", "2026-02-01",
                            database_path=self.database_path)
        duplicate_id = save_task(plot_id, " check SOIL ", "irrigation", date(2026, 2, 1),
                                 database_path=self.database_path)
        self.assertEqual(task_id, duplicate_id)
        save_task(plot_id, "Check soil", "irrigation", "2026-02-02",
                  "Updated note", task_id=task_id, database_path=self.database_path)
        set_task_status(task_id, "completed", self.database_path)
        task = list_tasks(self.database_path)[0]
        self.assertEqual(task["due_date"], "2026-02-02")
        self.assertEqual(task["notes"], "Updated note")
        self.assertEqual(task["status"], "completed")
        delete_task(task_id, self.database_path)
        self.assertEqual(list_tasks(self.database_path), [])

    def test_irrigation_log_and_plot_deletion(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        save_irrigation_log(plot_id, date(2026, 2, 1), database_path=self.database_path)
        save_irrigation_log(plot_id, "2026-02-03", database_path=self.database_path)
        self.assertEqual(latest_irrigation_date(plot_id, self.database_path), "2026-02-03")
        delete_plot(plot_id, self.database_path)
        self.assertIsNone(latest_irrigation_date(plot_id, self.database_path))

    def test_soil_test_saves_context_and_rejects_invalid_measurements(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        save_soil_test(
            plot_id, date(2026, 2, 3), {"ph": 6.5, "nitrogen": "12", "potassium": ""},
            "Lab report", crop="Tomato", growth_stage="Flowering", soil_type="Loam",
            database_path=self.database_path,
        )
        record = list_soil_tests(database_path=self.database_path)[0]
        self.assertEqual(record["ph"], 6.5)
        self.assertEqual(record["nitrogen"], 12)
        self.assertIsNone(record["potassium"])
        self.assertEqual(record["growth_stage"], "Flowering")
        with self.assertRaises(ValueError):
            save_soil_test(None, date.today(), {"ph": 15}, database_path=self.database_path)

    def test_financial_sales_expenses_filtering_and_profit_loss(self):
        save_financial_record(
            None, "expense", "labour", "Field work", "2026-02-01",
            amount="123.45", database_path=self.database_path,
        )
        save_financial_record(
            None, "sale", "harvest sale", "Tomato harvest", "2026-02-05",
            quantity_kg="12.5", selling_price_per_kg="40",
            database_path=self.database_path,
        )
        february_records = list_financial_records(
            "2026-02-01", "2026-02-28", database_path=self.database_path
        )
        summary = calculate_financial_summary(february_records)
        self.assertEqual(summary, {
            "expenses_paise": 12345,
            "income_paise": 50000,
            "profit_paise": 37655,
        })
        self.assertEqual(
            calculate_financial_summary([
                {"transaction_type": "expense", "amount_paise": 70000},
                {"transaction_type": "sale", "amount_paise": 50000},
            ])["profit_paise"],
            -20000,
        )
        self.assertEqual(
            list_financial_records("2026-03-01", "2026-03-31", database_path=self.database_path),
            [],
        )

    def test_soil_schema_upgrade_adds_context_columns(self):
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute(
                """CREATE TABLE soil_tests (
                    id INTEGER PRIMARY KEY,
                    plot_id INTEGER,
                    test_date TEXT NOT NULL,
                    ph REAL,
                    nitrogen REAL,
                    phosphorus REAL,
                    potassium REAL,
                    organic_matter REAL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )"""
            )
            connection.commit()
        finally:
            connection.close()
        initialize_database(self.database_path)
        connection = sqlite3.connect(self.database_path)
        try:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(soil_tests)")
            }
        finally:
            connection.close()
        self.assertTrue({"crop", "growth_stage", "soil_type"}.issubset(columns))

    def test_delete_local_records_clears_records_without_removing_schema(self):
        plot_id = save_plot(self.plot, database_path=self.database_path)
        save_scan(plot_id, [{"key": "Early_blight", "score": 0.8}], self.database_path)
        save_irrigation_log(plot_id, date.today(), database_path=self.database_path)
        save_task(plot_id, "Check crop", "crop task", date.today(), database_path=self.database_path)
        save_soil_test(plot_id, date.today(), {}, database_path=self.database_path)
        save_financial_record(
            plot_id, "expense", "seeds", "Seeds", date.today(), amount=10,
            database_path=self.database_path,
        )
        delete_local_records(self.database_path)
        self.assertEqual(list_plots(self.database_path), [])
        self.assertEqual(list_scans(self.database_path), [])
        self.assertEqual(list_tasks(self.database_path), [])
        self.assertEqual(list_soil_tests(database_path=self.database_path), [])
        self.assertEqual(list_financial_records(database_path=self.database_path), [])


if __name__ == "__main__":
    unittest.main()
