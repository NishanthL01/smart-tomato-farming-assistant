from collections.abc import Mapping


def normalize_class_key(class_key: object) -> str:
    key = str(class_key).strip()
    if key.startswith("Tomato___"):
        return key[len("Tomato___"):]
    return key


def find_treatment_entry(
    database: Mapping[str, object],
    class_key: str,
) -> Mapping[str, object] | None:
    normalized_key = normalize_class_key(class_key)

    for database_key, entry in database.items():
        if normalize_class_key(database_key) == normalized_key and isinstance(entry, Mapping):
            return entry

    return None


def get_treatment_options(entry: Mapping[str, object], option_key: str) -> list[Mapping[str, object]]:
    options = entry.get(option_key, [])
    if not isinstance(options, list):
        return []
    return [option for option in options if isinstance(option, Mapping)]
