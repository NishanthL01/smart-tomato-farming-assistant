import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Generator

DATA_DIR = Path(
    os.environ.get("TOMATO_APP_DATA_DIR", Path.home() / ".smart_tomato_farming")
)
DATABASE_PATH = DATA_DIR / "farm_records.sqlite3"

AREA_UNITS = {"cents", "acres", "hectares"}
CULTIVATION_TYPES = {"Open field", "Protected cultivation"}
IRRIGATION_METHODS = {"Not specified", "Drip", "Furrow", "Sprinkler", "Other"}
CROP_STAGES = {
    "Not set",
    "Nursery",
    "Vegetative",
    "Flowering",
    "Fruit development",
    "Harvest",
    "Completed",
}
TASK_CATEGORIES = {"irrigation", "crop task", "reminder"}
TASK_STATUSES = {"pending", "completed", "skipped"}
EXPENSE_CATEGORIES = {
    "seeds",
    "fertilizer",
    "pesticides",
    "irrigation",
    "labour",
    "transport",
    "miscellaneous",
}


@contextmanager
def _connect(database_path: Path = DATABASE_PATH) -> Generator[sqlite3.Connection, None, None]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(database_path: Path = DATABASE_PATH) -> None:
    with _connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS plots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_name TEXT NOT NULL,
                plot_name TEXT NOT NULL,
                village TEXT NOT NULL DEFAULT '',
                district TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT '',
                tomato_variety TEXT NOT NULL DEFAULT '',
                planting_date TEXT NOT NULL,
                area REAL NOT NULL CHECK (area > 0),
                area_unit TEXT NOT NULL,
                soil_type TEXT NOT NULL DEFAULT '',
                cultivation_type TEXT NOT NULL,
                irrigation_method TEXT NOT NULL,
                crop_stage TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER REFERENCES plots(id) ON DELETE SET NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                alternatives_json TEXT NOT NULL,
                farmer_confirmed_label TEXT,
                created_at TEXT NOT NULL,
                confirmed_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS irrigation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER NOT NULL REFERENCES plots(id) ON DELETE CASCADE,
                irrigation_date TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farm_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER REFERENCES plots(id) ON DELETE SET NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                due_date TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                dedupe_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS soil_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER REFERENCES plots(id) ON DELETE SET NULL,
                crop TEXT NOT NULL DEFAULT '',
                growth_stage TEXT NOT NULL DEFAULT '',
                soil_type TEXT NOT NULL DEFAULT '',
                test_date TEXT NOT NULL,
                ph REAL,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                organic_matter REAL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        soil_test_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(soil_tests)").fetchall()
        }
        for column in ("crop", "growth_stage", "soil_type"):
            if column not in soil_test_columns:
                connection.execute(
                    f"ALTER TABLE soil_tests ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS farm_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plot_id INTEGER REFERENCES plots(id) ON DELETE SET NULL,
                transaction_type TEXT NOT NULL CHECK (transaction_type IN ('expense', 'sale')),
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                transaction_date TEXT NOT NULL,
                amount_paise INTEGER NOT NULL CHECK (amount_paise >= 0),
                quantity_kg REAL,
                selling_price_per_kg REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agricultural_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                issuing_organization TEXT NOT NULL,
                crop TEXT NOT NULL,
                region TEXT NOT NULL,
                publication_date TEXT,
                review_date TEXT,
                source_url TEXT NOT NULL,
                verification_status TEXT NOT NULL DEFAULT 'unverified'
                    CHECK (verification_status IN ('unverified', 'verified', 'retired')),
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def validate_plot(data: dict[str, Any]) -> dict[str, Any]:
    farm_name = str(data.get("farm_name", "")).strip()
    plot_name = str(data.get("plot_name", "")).strip()
    if not farm_name or not plot_name:
        raise ValueError("Farm name and plot name are required.")

    planting_date = data.get("planting_date")
    if isinstance(planting_date, date):
        planting_date = planting_date.isoformat()
    try:
        planting_date = date.fromisoformat(str(planting_date)).isoformat()
    except ValueError as error:
        raise ValueError("Enter a valid planting date.") from error

    try:
        area = float(data.get("area"))
    except (TypeError, ValueError) as error:
        raise ValueError("Plot area must be a number greater than zero.") from error
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Plot area must be a number greater than zero.")

    area_unit = str(data.get("area_unit", ""))
    cultivation_type = str(data.get("cultivation_type", ""))
    irrigation_method = str(data.get("irrigation_method", ""))
    crop_stage = str(data.get("crop_stage", ""))
    if area_unit not in AREA_UNITS:
        raise ValueError("Choose a supported plot-area unit.")
    if cultivation_type not in CULTIVATION_TYPES:
        raise ValueError("Choose a supported cultivation type.")
    if irrigation_method not in IRRIGATION_METHODS:
        raise ValueError("Choose a supported irrigation method.")
    if crop_stage not in CROP_STAGES:
        raise ValueError("Choose a supported crop stage.")

    return {
        "farm_name": farm_name,
        "plot_name": plot_name,
        "village": str(data.get("village", "")).strip(),
        "district": str(data.get("district", "")).strip(),
        "state": str(data.get("state", "")).strip(),
        "tomato_variety": str(data.get("tomato_variety", "")).strip(),
        "planting_date": planting_date,
        "area": area,
        "area_unit": area_unit,
        "soil_type": str(data.get("soil_type", "")).strip(),
        "cultivation_type": cultivation_type,
        "irrigation_method": irrigation_method,
        "crop_stage": crop_stage,
        "notes": str(data.get("notes", "")).strip(),
    }


def list_plots(database_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM plots ORDER BY farm_name, plot_name, id"
        ).fetchall()
    return [dict(row) for row in rows]


def save_plot(
    data: dict[str, Any],
    plot_id: int | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    plot = validate_plot(data)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    columns = (
        "farm_name", "plot_name", "village", "district", "state", "tomato_variety",
        "planting_date", "area", "area_unit", "soil_type", "cultivation_type",
        "irrigation_method", "crop_stage", "notes",
    )
    values = tuple(plot[column] for column in columns)
    initialize_database(database_path)
    with _connect(database_path) as connection:
        if plot_id is None:
            placeholders = ", ".join("?" for _ in columns)
            column_sql = ", ".join(columns)
            cursor = connection.execute(
                f"INSERT INTO plots ({column_sql}, created_at, updated_at) "
                f"VALUES ({placeholders}, ?, ?)",
                (*values, now, now),
            )
            return int(cursor.lastrowid)

        assignments = ", ".join(f"{column} = ?" for column in columns)
        cursor = connection.execute(
            f"UPDATE plots SET {assignments}, updated_at = ? WHERE id = ?",
            (*values, now, plot_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected plot no longer exists.")
    return plot_id


def delete_plot(plot_id: int, database_path: Path = DATABASE_PATH) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM plots WHERE id = ?", (plot_id,))
        if cursor.rowcount != 1:
            raise ValueError("The selected plot no longer exists.")


def save_scan(
    plot_id: int | None,
    results: list[dict[str, Any]],
    database_path: Path = DATABASE_PATH,
) -> int:
    if not results:
        raise ValueError("A scan must contain at least one prediction.")
    predicted_label = str(results[0].get("key", "")).strip()
    if not predicted_label:
        raise ValueError("The scan prediction label is required.")

    confidence = float(results[0].get("score", float("nan")))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("The scan confidence must be between zero and one.")

    alternatives = []
    for result in results[1:3]:
        label = str(result.get("key", "")).strip()
        score = float(result.get("score", float("nan")))
        if not label or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Alternative predictions contain invalid values.")
        alternatives.append({"label": label, "score": score})

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO scans (
                plot_id, predicted_label, confidence, alternatives_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (plot_id, predicted_label, confidence, json.dumps(alternatives), now),
        )
        return int(cursor.lastrowid)


def list_scans(database_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT scans.*, plots.farm_name, plots.plot_name
            FROM scans
            LEFT JOIN plots ON plots.id = scans.plot_id
            ORDER BY scans.created_at DESC, scans.id DESC
            """
        ).fetchall()

    scans = []
    for row in rows:
        scan = dict(row)
        scan["alternatives"] = json.loads(scan.pop("alternatives_json"))
        scans.append(scan)
    return scans


def confirm_scan(
    scan_id: int,
    confirmed_label: str | None,
    database_path: Path = DATABASE_PATH,
) -> None:
    label = str(confirmed_label).strip() if confirmed_label is not None else None
    if label == "":
        label = None
    confirmed_at = datetime.now().astimezone().isoformat(timespec="seconds") if label else None
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE scans
            SET farmer_confirmed_label = ?, confirmed_at = ?
            WHERE id = ?
            """,
            (label, confirmed_at, scan_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected scan no longer exists.")


def delete_scan(scan_id: int, database_path: Path = DATABASE_PATH) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        if cursor.rowcount != 1:
            raise ValueError("The selected scan no longer exists.")


def _valid_iso_date(value: date | str, field_name: str) -> str:
    try:
        return date.fromisoformat(value.isoformat() if isinstance(value, date) else str(value)).isoformat()
    except ValueError as error:
        raise ValueError(f"Enter a valid {field_name}.") from error


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_irrigation_log(
    plot_id: int,
    irrigation_date: date | str,
    notes: str = "",
    database_path: Path = DATABASE_PATH,
) -> int:
    validated_date = _valid_iso_date(irrigation_date, "irrigation date")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO irrigation_logs (plot_id, irrigation_date, notes, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (plot_id, validated_date, str(notes).strip(), _timestamp()),
        )
        return int(cursor.lastrowid)


def latest_irrigation_date(
    plot_id: int,
    database_path: Path = DATABASE_PATH,
) -> str | None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT irrigation_date FROM irrigation_logs
            WHERE plot_id = ? ORDER BY irrigation_date DESC, id DESC LIMIT 1
            """,
            (plot_id,),
        ).fetchone()
    return str(row["irrigation_date"]) if row else None


def save_task(
    plot_id: int | None,
    title: str,
    category: str,
    due_date: date | str,
    notes: str = "",
    task_id: int | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    clean_title = str(title).strip()
    if not clean_title:
        raise ValueError("A task title is required.")
    if category not in TASK_CATEGORIES:
        raise ValueError("Choose a supported task category.")
    validated_date = _valid_iso_date(due_date, "task date")
    dedupe_key = f"{plot_id or 0}|{category.casefold()}|{validated_date}|{clean_title.casefold()}"
    now = _timestamp()
    initialize_database(database_path)
    with _connect(database_path) as connection:
        if task_id is None:
            existing = connection.execute(
                "SELECT id FROM farm_tasks WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = connection.execute(
                """
                INSERT INTO farm_tasks (
                    plot_id, title, category, due_date, notes, dedupe_key,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plot_id,
                    clean_title,
                    category,
                    validated_date,
                    str(notes).strip(),
                    dedupe_key,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

        cursor = connection.execute(
            """
            UPDATE farm_tasks SET plot_id = ?, title = ?, category = ?, due_date = ?,
                notes = ?, dedupe_key = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                plot_id,
                clean_title,
                category,
                validated_date,
                str(notes).strip(),
                dedupe_key,
                now,
                task_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected task no longer exists.")
    return task_id


def list_tasks(
    database_path: Path = DATABASE_PATH,
    include_completed: bool = True,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    sql = """
        SELECT farm_tasks.*, plots.farm_name, plots.plot_name
        FROM farm_tasks LEFT JOIN plots ON plots.id = farm_tasks.plot_id
    """
    if not include_completed:
        sql += " WHERE farm_tasks.status = 'pending'"
    sql += " ORDER BY farm_tasks.due_date, farm_tasks.id"
    with _connect(database_path) as connection:
        rows = connection.execute(sql).fetchall()
    return [dict(row) for row in rows]


def set_task_status(
    task_id: int,
    status: str,
    database_path: Path = DATABASE_PATH,
) -> None:
    if status not in TASK_STATUSES:
        raise ValueError("Choose a supported task status.")
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "UPDATE farm_tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, _timestamp(), task_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("The selected task no longer exists.")


def delete_task(task_id: int, database_path: Path = DATABASE_PATH) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute("DELETE FROM farm_tasks WHERE id = ?", (task_id,))
        if cursor.rowcount != 1:
            raise ValueError("The selected task no longer exists.")


def save_soil_test(
    plot_id: int | None,
    test_date: date | str,
    results: dict[str, Any],
    notes: str = "",
    database_path: Path = DATABASE_PATH,
    crop: str = "Tomato",
    growth_stage: str = "",
    soil_type: str = "",
) -> int:
    validated_date = _valid_iso_date(test_date, "soil-test date")
    nutrient_values: dict[str, float | None] = {}
    for field in ("ph", "nitrogen", "phosphorus", "potassium", "organic_matter"):
        raw_value = results.get(field)
        if raw_value in (None, ""):
            nutrient_values[field] = None
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number.")
        if field == "ph" and not 0 <= value <= 14:
            raise ValueError("pH must be between 0 and 14.")
        if field != "ph" and value < 0:
            raise ValueError(f"{field} cannot be negative.")
        nutrient_values[field] = value

    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO soil_tests (
                plot_id, crop, growth_stage, soil_type, test_date, ph, nitrogen, phosphorus, potassium,
                organic_matter, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plot_id,
                str(crop).strip(),
                str(growth_stage).strip(),
                str(soil_type).strip(),
                validated_date,
                nutrient_values["ph"],
                nutrient_values["nitrogen"],
                nutrient_values["phosphorus"],
                nutrient_values["potassium"],
                nutrient_values["organic_matter"],
                str(notes).strip(),
                _timestamp(),
            ),
        )
        return int(cursor.lastrowid)


def list_soil_tests(
    plot_id: int | None = None,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    initialize_database(database_path)
    sql = """
        SELECT soil_tests.*, plots.farm_name, plots.plot_name
        FROM soil_tests LEFT JOIN plots ON plots.id = soil_tests.plot_id
    """
    parameters: tuple[Any, ...] = ()
    if plot_id is not None:
        sql += " WHERE soil_tests.plot_id = ?"
        parameters = (plot_id,)
    sql += " ORDER BY soil_tests.test_date DESC, soil_tests.id DESC"
    with _connect(database_path) as connection:
        rows = connection.execute(sql, parameters).fetchall()
    return [dict(row) for row in rows]


def save_financial_record(
    plot_id: int | None,
    transaction_type: str,
    category: str,
    description: str,
    transaction_date: date | str,
    amount: int | float | str | Decimal | None = None,
    quantity_kg: int | float | str | Decimal | None = None,
    selling_price_per_kg: int | float | str | Decimal | None = None,
    database_path: Path = DATABASE_PATH,
) -> int:
    if transaction_type not in {"expense", "sale"}:
        raise ValueError("Choose expense or sale.")
    if transaction_type == "expense" and category not in EXPENSE_CATEGORIES:
        raise ValueError("Choose a supported expense category.")
    clean_description = str(description).strip()
    if not clean_description:
        raise ValueError("A record description is required.")
    validated_date = _valid_iso_date(transaction_date, "transaction date")

    def money(value: Any, field_name: str) -> Decimal:
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must be a valid amount.") from error
        if not parsed.is_finite() or parsed < 0:
            raise ValueError(f"{field_name} must be a finite non-negative amount.")
        return parsed

    quantity_value = None
    price_value = None
    if transaction_type == "sale":
        quantity_value = money(quantity_kg, "Harvest quantity")
        price_value = money(selling_price_per_kg, "Selling price")
        amount_decimal = quantity_value * price_value
        if quantity_value <= 0:
            raise ValueError("Harvest quantity must be greater than zero.")
        category = "harvest sale"
    else:
        amount_decimal = money(amount, "Expense amount")

    amount_paise = int((amount_decimal * 100).quantize(Decimal("1")))
    initialize_database(database_path)
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO farm_transactions (
                plot_id, transaction_type, category, description, transaction_date,
                amount_paise, quantity_kg, selling_price_per_kg, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plot_id,
                transaction_type,
                category,
                clean_description,
                validated_date,
                amount_paise,
                float(quantity_value) if quantity_value is not None else None,
                float(price_value) if price_value is not None else None,
                _timestamp(),
            ),
        )
        return int(cursor.lastrowid)


def list_financial_records(
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    plot_id: int | None = None,
    database_path: Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    conditions = []
    parameters: list[Any] = []
    if start_date is not None:
        conditions.append("transaction_date >= ?")
        parameters.append(_valid_iso_date(start_date, "start date"))
    if end_date is not None:
        conditions.append("transaction_date <= ?")
        parameters.append(_valid_iso_date(end_date, "end date"))
    if plot_id is not None:
        conditions.append("plot_id = ?")
        parameters.append(plot_id)

    sql = """
        SELECT farm_transactions.*, plots.farm_name, plots.plot_name
        FROM farm_transactions
        LEFT JOIN plots ON plots.id = farm_transactions.plot_id
    """
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY transaction_date, id"
    initialize_database(database_path)
    with _connect(database_path) as connection:
        rows = connection.execute(sql, parameters).fetchall()
    return [dict(row) for row in rows]


def calculate_financial_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    expenses = sum(
        int(record["amount_paise"])
        for record in records
        if record["transaction_type"] == "expense"
    )
    income = sum(
        int(record["amount_paise"])
        for record in records
        if record["transaction_type"] == "sale"
    )
    return {"expenses_paise": expenses, "income_paise": income, "profit_paise": income - expenses}


def delete_local_records(database_path: Path = DATABASE_PATH) -> None:
    initialize_database(database_path)
    with _connect(database_path) as connection:
        for table in (
            "irrigation_logs",
            "farm_tasks",
            "soil_tests",
            "farm_transactions",
            "scans",
            "plots",
            "agricultural_references",
        ):
            connection.execute(f"DELETE FROM {table}")
