from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".parquet", ".h5", ".hdf5", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl"}
FORBIDDEN_IDENTIFIER_COLUMNS = {
    "subject_id",
    "stay_id",
    "hadm_id",
    "patient_id",
    "person_id",
    "mrn",
}
FORBIDDEN_PATTERNS = {
    "personal path": re.compile(r"/(?:scratch2?|home)/(?:luisabarros|leticia\.ribeiro)(?:/|$)", re.I),
    "database password value": re.compile(r'(?:password|passwd)\s*[=:]\s*["\'][^"\']+', re.I),
    "database URI with credentials": re.compile(r"postgres(?:ql)?://[^:@/\s]+:[^@/\s]+@", re.I),
    "interactive credential prompt": re.compile(r"\bgetpass\s*\(", re.I),
    "PostgreSQL password environment": re.compile(r"\bPGPASSWORD\b", re.I),
    "internal database identity": re.compile(r"\buti_user\b|MIMIC_DB_(?:HOST|PORT|NAME|USER|PASSWORD)", re.I),
    "explicit PostgreSQL connection parameters": re.compile(
        r"psycopg2\.connect\s*\([^)]*\b(?:host|port|dbname|user|password)\s*=",
        re.I | re.S,
    ),
}

errors = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    relative = path.relative_to(ROOT)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden file type: {relative}")
    if path.stat().st_size > 5 * 1024 * 1024:
        errors.append(f"file larger than 5 MiB: {relative}")
    if any(token in path.name.lower() for token in FORBIDDEN_IDENTIFIER_COLUMNS):
        errors.append(f"identifier-bearing filename: {relative}")
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            header = next(csv.reader(handle, delimiter=delimiter), [])
        identifier_columns = FORBIDDEN_IDENTIFIER_COLUMNS.intersection(
            column.strip().lower() for column in header
        )
        if identifier_columns:
            errors.append(
                f"identifier column(s) {sorted(identifier_columns)}: {relative}"
            )
    if path.suffix.lower() in {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".example", ".ipynb"} or path.name in {".gitignore"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if relative != Path("scripts/check_public_release.py"):
            for label, pattern in FORBIDDEN_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{label}: {relative}")
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
                errors.append(f"stored notebook output: {relative}, cell {index}")

if errors:
    print("PUBLIC RELEASE CHECK FAILED")
    print("\n".join(f"- {item}" for item in errors))
    sys.exit(1)
print("PUBLIC RELEASE CHECK PASSED")
