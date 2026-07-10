from pathlib import Path

import pandas as pd


FEDKIW_TEST_NAMES = {
    "test1": "FedkiwA",
    "test2": "FedkiwB",
    "test3": "FedkiwC",
    "test4": "FedkiwD1",
    "test5": "FedkiwD2",
}

FEDKIW_NAME_TO_TEST = {
    value.lower(): key
    for key, value in FEDKIW_TEST_NAMES.items()
}

EXACT_FIELD_NAMES = {
    "density": "rho",
    "rho": "rho",
    "velocity": "u0",
    "u": "u0",
    "u0": "u0",
    "pressure": "p",
    "p": "p",
    "entropy": "entropy",
    "specific internal energy": "e",
    "internal energy": "e",
    "energy": "e",
}


def infer_fedkiw_test_name(text):
    lower = str(text).lower()

    for test_name in FEDKIW_TEST_NAMES:
        if test_name in lower:
            return test_name

    for fedkiw_name, test_name in FEDKIW_NAME_TO_TEST.items():
        if fedkiw_name in lower:
            return test_name

    return None


def exact_reference_path(exact_root, test_name):
    if test_name is None:
        return None

    return Path(exact_root) / f"{test_name}_exact.csv"


def load_digitized_exact_reference(csv_path):
    raw = pd.read_csv(csv_path, header=None)
    fields = {}

    for column in range(0, raw.shape[1] - 1, 2):
        name = str(raw.iat[0, column]).strip().lower()
        field_key = EXACT_FIELD_NAMES.get(name)

        if field_key is None:
            continue

        x = pd.to_numeric(raw.iloc[2:, column], errors="coerce")
        y = pd.to_numeric(raw.iloc[2:, column + 1], errors="coerce")
        valid = x.notna() & y.notna()

        if valid.any():
            fields[field_key] = (
                x[valid].to_numpy(),
                y[valid].to_numpy(),
            )

    return fields


def load_optional_exact_reference(exact_root, test_name, context="plot"):
    path = exact_reference_path(exact_root, test_name)

    if path is None:
        return None

    if not path.exists():
        print(f"[exact-reference] {context}: {path} not found; continuing without exact overlay")
        return None

    fields = load_digitized_exact_reference(path)

    if not fields:
        print(f"[exact-reference] {context}: {path} did not contain recognised exact fields")
        return None

    return fields


def find_exact_reference_for_context(exact_root, context_text):
    exact_root = Path(exact_root)
    lower_context = str(context_text).lower()

    test_name = infer_fedkiw_test_name(lower_context)
    if test_name is not None:
        return exact_root / "fedkiw" / f"{test_name}_exact.csv"

    for csv_path in sorted(exact_root.rglob("*_exact.csv")):
        case_name = csv_path.stem[:-len("_exact")].lower()
        if case_name and case_name in lower_context:
            return csv_path

    return None


def load_optional_exact_reference_for_context(exact_root, context_text, context="plot"):
    path = find_exact_reference_for_context(exact_root, context_text)

    if path is None:
        return None

    if not path.exists():
        print(f"[exact-reference] {context}: {path} not found; continuing without exact overlay")
        return None

    fields = load_digitized_exact_reference(path)

    if not fields:
        print(f"[exact-reference] {context}: {path} did not contain recognised exact fields")
        return None

    return fields
