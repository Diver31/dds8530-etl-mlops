from pathlib import Path
import json
import time

import dask
import dask.dataframe as dd
import pandas as pd

SOURCE = Path("data/processed/merchandise_sales.csv")
PROFILE = Path("reports/source_profile.json")
CLASSIFICATION = Path(
    "reports/merchandise_classification_audit.json"
)
OUTPUT = Path("data/processed/weekly_product_sales.csv")
AUDIT = Path("reports/weekly_aggregation_audit.json")

profile = json.loads(PROFILE.read_text())
classification = json.loads(CLASSIFICATION.read_text())

expected = classification["classification_counts"][
    "merchandise_candidate"
]

def aggregate_partition(frame):
    dates = pd.to_datetime(
        frame["InvoiceDate"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    quantities = pd.to_numeric(
        frame["Quantity"], errors="raise"
    )

    if quantities.isna().any() or (quantities <= 0).any():
        raise ValueError("Invalid merchandise quantity.")

    # Monday is the start of each reporting week.
    monday = (
        dates.dt.normalize()
        - pd.to_timedelta(dates.dt.weekday, unit="D")
    )

    prepared = pd.DataFrame({
        "StockCode": frame["StockCode"].astype("string"),
        "week_start": monday,
        "units_sold": quantities.astype("int64"),
    })

    return prepared.groupby(
        ["StockCode", "week_start"],
        as_index=False,
        sort=False,
    )["units_sold"].sum()


meta = pd.DataFrame({
    "StockCode": pd.Series(dtype="string"),
    "week_start": pd.Series(dtype="datetime64[ns]"),
    "units_sold": pd.Series(dtype="int64"),
})

started = time.perf_counter()

df = dd.read_csv(
    SOURCE,
    usecols=["StockCode", "InvoiceDate", "Quantity"],
    dtype={
        "StockCode": "string",
        "InvoiceDate": "string",
        "Quantity": "int64",
    },
    blocksize=16_000_000,
)

partition_count = df.npartitions

partition_totals = df.map_partitions(
    aggregate_partition, meta=meta
)

# Dask combines matching product-weeks across partitions.
weekly_task = partition_totals.groupby(
    ["StockCode", "week_start"]
)["units_sold"].sum(split_out=4)

weekly_series, source_rows, source_units = dask.compute(
    weekly_task,
    df.map_partitions(len).sum(),
    df["Quantity"].sum(),
)

weekly = weekly_series.reset_index()
weekly = weekly.sort_values(
    ["StockCode", "week_start"]
).reset_index(drop=True)

# Reconcile aggregation before removing partial weeks.
if int(source_rows) != expected:
    raise RuntimeError("Source row count mismatch.")

if int(weekly["units_sold"].sum()) != int(source_units):
    raise RuntimeError("Quantity reconciliation failed.")

# The source begins and ends within incomplete weeks.
first_date = pd.Timestamp(
    profile["earliest_invoice"]
).normalize()

last_date = pd.Timestamp(
    profile["latest_invoice"]
).normalize()

first_full_week = (
    first_date
    - pd.Timedelta(days=first_date.weekday())
    + pd.Timedelta(days=7)
)

last_full_week = (
    last_date
    - pd.Timedelta(days=last_date.weekday())
    - pd.Timedelta(days=7)
)

complete = weekly.loc[
    weekly["week_start"].between(
        first_full_week, last_full_week
    )
].copy()

complete.to_csv(
    OUTPUT,
    index=False,
    date_format="%Y-%m-%d",
)

elapsed = time.perf_counter() - started

audit = {
    "input_rows": int(source_rows),
    "dask_partitions": partition_count,
    "input_units": int(source_units),
    "all_week_quantity_reconciliation": "PASS",
    "first_complete_week": str(first_full_week.date()),
    "last_complete_week": str(last_full_week.date()),
    "complete_calendar_weeks": len(
        pd.date_range(
            first_full_week,
            last_full_week,
            freq="W-MON",
        )
    ),
    "observed_product_week_rows": len(complete),
    "distinct_products": int(
        complete["StockCode"].nunique()
    ),
    "complete_week_units": int(
        complete["units_sold"].sum()
    ),
    "processing_seconds": round(elapsed, 3),
    "dask_version": dask.__version__,
    "pandas_version": pd.__version__,
    "missing_product_weeks":
        "Not yet zero-filled; required before model training",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== DASK WEEKLY AGGREGATION ===")
print("Source merchandise rows:", source_rows)
print("Dask partitions:", partition_count)
print("Total input units:", source_units)
print("All-week quantity reconciliation: PASS")
print("First complete week:", first_full_week.date())
print("Last complete week:", last_full_week.date())
print("Complete calendar weeks:",
      audit["complete_calendar_weeks"])
print("Observed product-week records:", len(complete))
print("Distinct products:", audit["distinct_products"])
print("Complete-week units:", audit["complete_week_units"])
print("Processing time (seconds):", round(elapsed, 3))
print("Output:", OUTPUT)
print("Audit:", AUDIT)
