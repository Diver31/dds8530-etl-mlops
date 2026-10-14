from pathlib import Path
import json

import dask
import dask.dataframe as dd
import pandas as pd

SOURCE = Path("data/processed/merchandise_sales.csv")
GLOBAL = Path("data/processed/weekly_product_sales.csv")
OUTPUT = Path("data/processed/uk_weekly_sales_observed.csv")
AUDIT = Path("reports/uk_weekly_audit.json")

source_audit = json.loads(
    Path("reports/merchandise_classification_audit.json").read_text()
)
weekly_audit = json.loads(
    Path("reports/weekly_aggregation_audit.json").read_text()
)
model_audit = json.loads(
    Path("reports/model_source_assessment.json").read_text()
)

start = pd.Timestamp(weekly_audit["first_complete_week"])
end = pd.Timestamp(weekly_audit["last_complete_week"])

meta = pd.DataFrame({
    "StockCode": pd.Series(dtype="string"),
    "week_start": pd.Series(dtype="datetime64[ns]"),
    "units_sold": pd.Series(dtype="int64"),
    "transaction_rows": pd.Series(dtype="int64"),
})

def aggregate_partition(frame):
    uk = frame.loc[
        frame["Country"].str.strip() == "United Kingdom"
    ].copy()

    if uk.empty:
        return meta.copy()

    dates = pd.to_datetime(
        uk["InvoiceDate"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    quantities = uk["Quantity"].astype("int64")

    if (quantities <= 0).any():
        raise ValueError("Unexpected nonpositive quantity.")

    week_start = (
        dates.dt.normalize()
        - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    )

    prepared = pd.DataFrame({
        "StockCode": uk["StockCode"].str.strip().str.upper(),
        "week_start": week_start,
        "units_sold": quantities,
        "transaction_rows": 1,
    })

    return prepared.groupby(
        ["StockCode", "week_start"],
        as_index=False,
    ).agg({
        "units_sold": "sum",
        "transaction_rows": "sum",
    })

df = dd.read_csv(
    SOURCE,
    usecols=[
        "StockCode", "Country", "InvoiceDate", "Quantity"
    ],
    dtype={
        "StockCode": "string",
        "Country": "string",
        "InvoiceDate": "string",
        "Quantity": "int64",
    },
    blocksize=16_000_000,
)

uk_mask = df["Country"].str.strip() == "United Kingdom"

partial = df.map_partitions(
    aggregate_partition, meta=meta
)

grouped = partial.groupby(
    ["StockCode", "week_start"]
).agg({
    "units_sold": "sum",
    "transaction_rows": "sum",
}, split_out=4)

weekly, source_rows, uk_rows, uk_units = dask.compute(
    grouped,
    df.map_partitions(len).sum(),
    uk_mask.sum(),
    df["Quantity"].where(uk_mask, other=0).sum(),
)

weekly = weekly.reset_index()

if int(source_rows) != source_audit[
    "classification_counts"
]["merchandise_candidate"]:
    raise RuntimeError("Source row reconciliation failed.")

if int(uk_rows) != model_audit["uk_merchandise_rows"]:
    raise RuntimeError("UK transaction count mismatch.")

if int(weekly["units_sold"].sum()) != int(uk_units):
    raise RuntimeError("UK quantity reconciliation failed.")

# Retain only the same 52 complete weeks as the global data.
complete = weekly.loc[
    weekly["week_start"].between(start, end)
].copy()

complete = complete.sort_values(
    ["StockCode", "week_start"]
).reset_index(drop=True)

if complete.duplicated(
    ["StockCode", "week_start"]
).any():
    raise RuntimeError("Duplicate UK product-week keys.")

# Verify UK sales do not exceed global sales for any product-week.
global_weekly = pd.read_csv(
    GLOBAL,
    dtype={"StockCode": "string", "units_sold": "int64"},
    parse_dates=["week_start"],
)

matched = complete.merge(
    global_weekly,
    on=["StockCode", "week_start"],
    how="left",
    suffixes=("_uk", "_global"),
    validate="one_to_one",
)

if matched["units_sold_global"].isna().any():
    raise RuntimeError("UK product-week missing from global data.")

if (
    matched["units_sold_uk"] > matched["units_sold_global"]
).any():
    raise RuntimeError("UK quantities exceed global quantities.")

complete.to_csv(
    OUTPUT, index=False, date_format="%Y-%m-%d"
)

# Assess historical coverage without inventing zero-sales records.
coverage = complete.groupby("StockCode")[
    "week_start"
].nunique()

audit = {
    "source_merchandise_rows": int(source_rows),
    "uk_transaction_rows": int(uk_rows),
    "uk_units_all_dates": int(uk_units),
    "dask_partitions": df.npartitions,
    "first_complete_week": str(start.date()),
    "last_complete_week": str(end.date()),
    "uk_product_week_rows": len(complete),
    "uk_products_complete_weeks": int(
        complete["StockCode"].nunique()
    ),
    "uk_units_complete_weeks": int(
        complete["units_sold"].sum()
    ),
    "products_with_26_plus_observed_weeks": int(
        (coverage >= 26).sum()
    ),
    "products_with_40_plus_observed_weeks": int(
        (coverage >= 40).sum()
    ),
    "source_reconciliation": "PASS",
    "global_comparison": "PASS",
    "zero_sales_weeks": "Not yet generated",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== UK WEEKLY DATASET ===")
print("UK input transactions:", uk_rows)
print("UK units (all dates):", uk_units)
print("Dask partitions:", df.npartitions)
print("Complete-week product rows:", len(complete))
print("Products in complete weeks:",
      audit["uk_products_complete_weeks"])
print("UK units in complete weeks:",
      audit["uk_units_complete_weeks"])
print("Products with 26+ observed weeks:",
      audit["products_with_26_plus_observed_weeks"])
print("Products with 40+ observed weeks:",
      audit["products_with_40_plus_observed_weeks"])
print("Global comparison: PASS")
print("Source reconciliation: PASS")
print("Output:", OUTPUT)
print("Audit:", AUDIT)
