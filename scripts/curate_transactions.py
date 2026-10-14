from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import pandas as pd

INPUT = Path("data/staging/online_retail.csv")
PROFILE = Path("reports/source_profile.json")
OUTPUT = Path("data/processed/retail_sales_candidates.csv")
AUDIT = Path("reports/transaction_curation_audit.json")

# Verify source data has not changed since initial profiling.
profile = json.loads(PROFILE.read_text())

with INPUT.open("rb") as f:
    source_hash = sha256(f.read()).hexdigest()

if source_hash != profile["csv_sha256"]:
    raise RuntimeError("Source fingerprint mismatch.")

flags_total = Counter()
nonstandard_codes = Counter()
rows_read = 0
rows_saved = 0
missing_customer_kept = 0
missing_description_kept = 0

dtype_map = {
    "InvoiceNo": "string",
    "StockCode": "string",
    "Description": "string",
    "CustomerID": "string",
    "Country": "string",
}

# Process records in batches to limit memory usage.
for chunk in pd.read_csv(
    INPUT, dtype=dtype_map, chunksize=100_000
):
    start = rows_read + 1
    rows_read += len(chunk)

    chunk.insert(
        0, "source_row_number",
        range(start, rows_read + 1)
    )

    for col in ("InvoiceNo", "StockCode",
                "Description", "Country"):
        chunk[col] = chunk[col].str.strip()

    chunk["InvoiceDate"] = pd.to_datetime(
        chunk["InvoiceDate"], errors="coerce"
    )
    chunk["Quantity"] = pd.to_numeric(
        chunk["Quantity"], errors="coerce"
    )
    chunk["UnitPrice"] = pd.to_numeric(
        chunk["UnitPrice"], errors="coerce"
    )

    flags = {
        "cancelled_invoice":
            chunk["InvoiceNo"].str.upper().str.startswith(
                "C", na=False
            ),
        "nonpositive_or_missing_quantity":
            chunk["Quantity"].isna() |
            (chunk["Quantity"] <= 0),
        "nonpositive_or_missing_unit_price":
            chunk["UnitPrice"].isna() |
            (chunk["UnitPrice"] <= 0),
        "invalid_invoice_date":
            chunk["InvoiceDate"].isna(),
        "missing_invoice_number":
            chunk["InvoiceNo"].isna() |
            chunk["InvoiceNo"].eq(""),
        "missing_stock_code":
            chunk["StockCode"].isna() |
            chunk["StockCode"].eq(""),
    }

    eligible = pd.Series(True, index=chunk.index)

    for name, flag in flags.items():
        flag = flag.fillna(True)
        flags_total[name] += int(flag.sum())
        eligible &= ~flag

    valid = chunk.loc[eligible].copy()

    # Extended line value, not net accounting revenue.
    valid["line_value"] = (
        valid["Quantity"] * valid["UnitPrice"]
    ).round(2)

    missing_customer_kept += int(
        valid["CustomerID"].isna().sum()
    )
    missing_description_kept += int(
        valid["Description"].isna().sum()
    )

    # Review unusual codes without excluding them.
    standard_code = valid["StockCode"].str.fullmatch(
        r"\d{5}[A-Za-z]?", na=False
    )
    unusual = valid.loc[~standard_code, "StockCode"]
    nonstandard_codes.update(unusual.value_counts().to_dict())

    rows_saved += len(valid)

    valid.to_csv(
        OUTPUT,
        mode="w" if start == 1 else "a",
        header=(start == 1),
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

# Independently verify the output row count.
written_rows = sum(
    len(batch)
    for batch in pd.read_csv(OUTPUT, chunksize=100_000)
)

if rows_read != profile["row_count"]:
    raise RuntimeError("Input row count does not match profile.")

if written_rows != rows_saved:
    raise RuntimeError("Output row count mismatch.")

with OUTPUT.open("rb") as f:
    output_hash = sha256(f.read()).hexdigest()

audit = {
    "input_sha256": source_hash,
    "output_sha256": output_hash,
    "input_rows": rows_read,
    "candidate_sales_rows": rows_saved,
    "excluded_rows": rows_read - rows_saved,
    "exclusion_flags_overlapping": dict(flags_total),
    "retained_missing_customer_id": missing_customer_kept,
    "retained_missing_description": missing_description_kept,
    "duplicate_policy": "Retained pending investigation",
    "nonstandard_codes_for_review":
        nonstandard_codes.most_common(10),
    "scope": "Candidate positive paid sales; non-merchandise codes not yet excluded",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== TRANSACTION CURATION AUDIT ===")
print("Input rows:", rows_read)
print("Candidate sales rows:", rows_saved)
print("Excluded rows:", rows_read - rows_saved)
print("Exclusion flags (may overlap):", dict(flags_total))
print("Retained missing customer IDs:", missing_customer_kept)
print("Retained missing descriptions:", missing_description_kept)
print("Nonstandard stock codes to review:",
      nonstandard_codes.most_common(10))
print("Source/output reconciliation: PASS")
print("Processed CSV:", OUTPUT)
print("Audit report:", AUDIT)
