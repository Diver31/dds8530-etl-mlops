from collections import Counter
from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

reports = Path("reports")
weekly_audit = json.loads(
    (reports / "weekly_aggregation_audit.json").read_text()
)
classification = json.loads(
    (reports / "merchandise_classification_audit.json").read_text()
)
holiday_audit = json.loads(
    (reports / "holiday_api_audit.json").read_text()
)

# Read and validate the PostgreSQL sales totals.
url = URL.create(
    "postgresql+psycopg",
    username=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host="127.0.0.1",
    port=55432,
    database=os.environ["POSTGRES_DB"],
)

engine = create_engine(url, pool_pre_ping=True)

with engine.connect() as conn:
    db = conn.execute(text("""
        SELECT
            COUNT(*) AS records,
            COUNT(DISTINCT stock_code) AS products,
            SUM(units_sold) AS units
        FROM etl.fact_weekly_sales
    """)).mappings().one()

engine.dispose()

db_rows = int(db["records"])
db_products = int(db["products"])
db_units = int(db["units"])

assert db_rows == weekly_audit["observed_product_week_rows"]
assert db_products == weekly_audit["distinct_products"]
assert db_units == weekly_audit["complete_week_units"]

print("=== POSTGRESQL EXTRACTION ===")
print("Records:", db_rows)
print("Products:", db_products)
print("Units:", db_units)
print("Database reconciliation: PASS")

# Inspect country coverage without changing source records.
country_rows = Counter()
country_units = Counter()
uk_products = set()

source_rows = 0

for batch in pd.read_csv(
    "data/processed/merchandise_sales.csv",
    usecols=["Country", "StockCode", "Quantity"],
    dtype={
        "Country": "string",
        "StockCode": "string",
        "Quantity": "int64",
    },
    chunksize=100_000,
):
    source_rows += len(batch)
    batch["Country"] = batch["Country"].fillna(
        "[MISSING]"
    ).str.strip()

    for country, group in batch.groupby("Country"):
        country_rows[country] += len(group)
        country_units[country] += int(
            group["Quantity"].sum()
        )

    uk = batch.loc[batch["Country"] == "United Kingdom"]
    uk_products.update(uk["StockCode"].dropna().unique())

expected = classification["classification_counts"][
    "merchandise_candidate"
]
assert source_rows == expected
assert sum(country_rows.values()) == source_rows

print("\n=== COUNTRY COVERAGE ===")
print("Total merchandise records:", source_rows)
print("Countries:", len(country_rows))

for country, count in country_rows.most_common(10):
    print(
        f"{country:<25} "
        f"Rows: {count:>8,} "
        f"Units: {country_units[country]:>10,}"
    )

uk_rows = country_rows["United Kingdom"]
uk_share = 100 * uk_rows / source_rows

print("\n=== UNITED KINGDOM ===")
print("UK merchandise records:", uk_rows)
print("UK share:", round(uk_share, 2), "%")
print("Distinct UK products:", len(uk_products))

# Validate UK-wide holiday coverage.
holidays = pd.read_csv(
    "data/staging/uk_holidays_2010_2011.csv"
)

assert len(holidays) == holiday_audit["holiday_records"]

holidays["holiday_date"] = pd.to_datetime(
    holidays["holiday_date"], errors="raise"
)

national = (
    holidays["is_national"]
    .astype("string")
    .str.lower()
    .map({"true": True, "false": False})
)

if national.isna().any():
    raise ValueError("Invalid holiday coverage flags.")

start = pd.Timestamp(weekly_audit["first_complete_week"])
end = (
    pd.Timestamp(weekly_audit["last_complete_week"])
    + pd.Timedelta(days=6)
)

included = holidays.loc[
    national
    & holidays["holiday_date"].between(start, end)
]

print("\n=== HOLIDAY MODELING COVERAGE ===")
print("Forecasting window:", start.date(), "to", end.date())
print("UK-wide holidays in window:", len(included))

for _, row in included.iterrows():
    print(row["holiday_date"].date(), row["holiday_name"])

result = {
    "database_rows": db_rows,
    "database_products": db_products,
    "database_units": db_units,
    "merchandise_rows": source_rows,
    "country_count": len(country_rows),
    "uk_merchandise_rows": uk_rows,
    "uk_share_percent": round(uk_share, 4),
    "uk_distinct_products": len(uk_products),
    "uk_wide_holidays_in_window": len(included),
    "validation": "PASS",
}

output = reports / "model_source_assessment.json"
output.write_text(json.dumps(result, indent=2))

print("\n=== FINAL VALIDATION ===")
print("SQL extraction: PASS")
print("Country reconciliation: PASS")
print("Holiday validation: PASS")
print("Report saved:", output)
