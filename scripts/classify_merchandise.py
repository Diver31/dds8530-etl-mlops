from collections import Counter
from pathlib import Path
import hashlib
import json
import pandas as pd

SOURCE = Path("data/processed/retail_sales_candidates.csv")
MERCH = Path("data/processed/merchandise_sales.csv")
REVIEW = Path("data/processed/held_or_excluded_sales.csv")
AUDIT = Path("reports/merchandise_classification_audit.json")

# Explicitly reviewed non-merchandise transaction codes.
EXCLUDE = {
    "POST": "postage",
    "DOT": "dotcom_postage",
    "C2": "carriage",
    "BANK CHARGES": "bank_charge",
    "AMAZONFEE": "platform_fee",
    "B": "bad_debt_adjustment",
    "S": "samples",
}

# Ambiguous records require further investigation.
HOLD = {
    "M": "unattributed_manual_transaction",
    "PADS": "unusual_low_value_product",
}

# Legitimate physical products identified during review.
REVIEWED_PRODUCTS = {
    "DCGSSGIRL",
    "DCGSSBOY",
    "DCGS0003",
    "DCGS0076",
    "DCGS0070",
    "DCGS0069",
    "DCGS0004",  # Verified shoulder bag
}

counts = Counter()
reasons = Counter()
held_codes = Counter()
input_rows = 0

# Start each run with clean output files.
for path in (MERCH, REVIEW):
    path.unlink(missing_ok=True)

merch_header = True
review_header = True

for chunk in pd.read_csv(
    SOURCE,
    chunksize=100_000,
    dtype={
        "StockCode": "string",
        "InvoiceNo": "string",
        "Description": "string",
        "CustomerID": "string",
        "Country": "string",
    },
):
    input_rows += len(chunk)

    codes = chunk["StockCode"].str.strip().str.upper()
    chunk["StockCode"] = codes

    # The pattern is a screening rule, not proof of
    # physical merchandise.
    standard = codes.str.fullmatch(
        r"\d{5}[A-Z]{0,3}", na=False
    )

    status = pd.Series(
        "hold_review", index=chunk.index, dtype="string"
    )
    reason = pd.Series(
        "unreviewed_stock_code",
        index=chunk.index,
        dtype="string",
    )

    merchandise = standard | codes.isin(REVIEWED_PRODUCTS)
    status.loc[merchandise] = "merchandise_candidate"
    reason.loc[merchandise] = "product_code_accepted"

    # Gift vouchers are not physical inventory demand.
    gift = codes.str.startswith("GIFT_", na=False)
    status.loc[gift] = "exclude_nonmerchandise"
    reason.loc[gift] = "gift_voucher"

    for code, label in EXCLUDE.items():
        mask = codes.eq(code).fillna(False)
        status.loc[mask] = "exclude_nonmerchandise"
        reason.loc[mask] = label

    for code, label in HOLD.items():
        mask = codes.eq(code).fillna(False)
        status.loc[mask] = "hold_review"
        reason.loc[mask] = label

    chunk["classification"] = status
    chunk["classification_reason"] = reason

    counts.update(status.value_counts().to_dict())
    reasons.update(reason.value_counts().to_dict())

    held = chunk.loc[status == "hold_review", "StockCode"]
    held_codes.update(
        held.fillna("[MISSING]").value_counts().to_dict()
    )

    good = chunk.loc[
        status == "merchandise_candidate"
    ].copy()

    other = chunk.loc[
        status != "merchandise_candidate"
    ].copy()

    if not good.empty:
        good.to_csv(
            MERCH,
            mode="w" if merch_header else "a",
            header=merch_header,
            index=False,
        )
        merch_header = False

    if not other.empty:
        other.to_csv(
            REVIEW,
            mode="w" if review_header else "a",
            header=review_header,
            index=False,
        )
        review_header = False

# Independently verify both outputs.
def count_csv(path):
    if not path.exists():
        return 0
    return sum(
        len(part)
        for part in pd.read_csv(path, chunksize=100_000)
    )

merch_rows = count_csv(MERCH)
review_rows = count_csv(REVIEW)

if merch_rows + review_rows != input_rows:
    raise RuntimeError("Input/output reconciliation failed.")

if merch_rows != counts["merchandise_candidate"]:
    raise RuntimeError("Merchandise count mismatch.")

if review_rows != (
    counts["exclude_nonmerchandise"] +
    counts["hold_review"]
):
    raise RuntimeError("Review count mismatch.")

with SOURCE.open("rb") as f:
    source_hash = hashlib.file_digest(
        f, "sha256"
    ).hexdigest()

audit = {
    "source_sha256": source_hash,
    "input_records": input_rows,
    "classification_counts": dict(counts),
    "classification_reasons": dict(reasons),
    "held_stock_codes": dict(held_codes),
    "reconciliation": "PASS",
    "duplicate_policy": "Original duplicates retained",
    "classification_scope":
        "Provisional physical-merchandise candidates",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== MERCHANDISE CLASSIFICATION ===")
print("Input records:", input_rows)
print("Merchandise candidates:", merch_rows)
print(
    "Excluded non-merchandise:",
    counts["exclude_nonmerchandise"]
)
print("Held for review:", counts["hold_review"])
print("\nHeld stock codes:", dict(held_codes))
print("\nClassification reasons:", dict(reasons))
print("\nInput/output reconciliation: PASS")
print("Merchandise file:", MERCH)
print("Review file:", REVIEW)
print("Audit report:", AUDIT)
