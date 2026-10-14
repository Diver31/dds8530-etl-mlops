from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger, Date, Text, create_engine, text
)
from sqlalchemy.engine import URL

load_dotenv()

SOURCE = Path("data/processed/weekly_product_sales.csv")
AUDIT_SOURCE = Path("reports/weekly_aggregation_audit.json")
AUDIT_OUTPUT = Path("reports/postgres_load_audit.json")

audit = json.loads(AUDIT_SOURCE.read_text())

# Read the previously validated weekly dataset.
df = pd.read_csv(
    SOURCE,
    dtype={
        "StockCode": "string",
        "units_sold": "int64",
    },
    parse_dates=["week_start"],
)

expected_rows = int(audit["observed_product_week_rows"])
expected_units = int(audit["complete_week_units"])
expected_products = int(audit["distinct_products"])

# Validate data before modifying the database.
if len(df) != expected_rows:
    raise ValueError("CSV row count does not match audit.")

if int(df["units_sold"].sum()) != expected_units:
    raise ValueError("CSV units do not match audit.")

if df["StockCode"].nunique() != expected_products:
    raise ValueError("Product count does not match audit.")

if df.isna().any().any():
    raise ValueError("Unexpected null values in weekly data.")

if (df["units_sold"] <= 0).any():
    raise ValueError("Nonpositive sales quantity detected.")

if df.duplicated(["StockCode", "week_start"]).any():
    raise ValueError("Duplicate product-week keys detected.")

if not df["week_start"].dt.dayofweek.eq(0).all():
    raise ValueError("Some reporting weeks do not start Monday.")

if str(df["week_start"].min().date()) != audit["first_complete_week"]:
    raise ValueError("Unexpected first reporting week.")

if str(df["week_start"].max().date()) != audit["last_complete_week"]:
    raise ValueError("Unexpected last reporting week.")

# Prepare SQL column names and database-native dates.
data = pd.DataFrame({
    "stock_code": df["StockCode"],
    "week_start": df["week_start"].dt.date,
    "units_sold": df["units_sold"],
})

with SOURCE.open("rb") as file:
    csv_hash = sha256(file.read()).hexdigest()

# Use the local credentials saved during PostgreSQL setup.
url = URL.create(
    "postgresql+psycopg",
    username=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host="127.0.0.1",
    port=55432,
    database=os.environ["POSTGRES_DB"],
)

engine = create_engine(url, pool_pre_ping=True)

def get_metrics(connection, table):
    return connection.execute(text(f"""
        SELECT
            COUNT(*) AS record_count,
            COUNT(DISTINCT stock_code) AS products,
            COALESCE(SUM(units_sold), 0) AS total_units,
            MIN(week_start) AS first_week,
            MAX(week_start) AS last_week
        FROM {table}
    """)).mappings().one()

def verify(metrics):
    if int(metrics["record_count"]) != expected_rows:
        raise RuntimeError("Database row count mismatch.")

    if int(metrics["products"]) != expected_products:
        raise RuntimeError("Database product count mismatch.")

    if int(metrics["total_units"]) != expected_units:
        raise RuntimeError("Database sales quantity mismatch.")

    if str(metrics["first_week"]) != audit["first_complete_week"]:
        raise RuntimeError("Database first week mismatch.")

    if str(metrics["last_week"]) != audit["last_complete_week"]:
        raise RuntimeError("Database last week mismatch.")

# One transaction protects the load from partial completion.
with engine.begin() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS etl"))

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS etl.fact_weekly_sales (
            stock_code TEXT NOT NULL,
            week_start DATE NOT NULL,
            units_sold BIGINT NOT NULL
                CHECK (units_sold > 0),
            loaded_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (stock_code, week_start)
        )
    """))

    # Stage the entire validated CSV first.
    data.to_sql(
        "weekly_sales_stage",
        con=conn,
        schema="etl",
        if_exists="replace",
        index=False,
        chunksize=5000,
        dtype={
            "stock_code": Text(),
            "week_start": Date(),
            "units_sold": BigInteger(),
        },
    )

    stage_metrics = get_metrics(
        conn, "etl.weekly_sales_stage"
    )
    verify(stage_metrics)

    # Atomically replace the current weekly snapshot.
    conn.execute(text(
        "DELETE FROM etl.fact_weekly_sales"
    ))

    conn.execute(text("""
        INSERT INTO etl.fact_weekly_sales
            (stock_code, week_start, units_sold)
        SELECT stock_code, week_start, units_sold
        FROM etl.weekly_sales_stage
    """))

    final_metrics = get_metrics(
        conn, "etl.fact_weekly_sales"
    )
    verify(final_metrics)

    # Remove temporary staging data after validation.
    conn.execute(text(
        "DROP TABLE etl.weekly_sales_stage"
    ))

# Reconnect to verify the committed database state.
with engine.connect() as conn:
    committed = get_metrics(conn, "etl.fact_weekly_sales")
    verify(committed)

result = {
    "status": "PASS",
    "loaded_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_csv_sha256": csv_hash,
    "database": os.environ["POSTGRES_DB"],
    "table": "etl.fact_weekly_sales",
    "records": int(committed["record_count"]),
    "products": int(committed["products"]),
    "total_units": int(committed["total_units"]),
    "first_week": str(committed["first_week"]),
    "last_week": str(committed["last_week"]),
    "unique_product_week_keys": True,
    "refresh_method": "Transactional full refresh",
    "reconciliation": "PASS",
}

AUDIT_OUTPUT.write_text(json.dumps(result, indent=2))

engine.dispose()

print("\n=== POSTGRESQL ETL LOAD ===")
print("Database:", result["database"])
print("Table:", result["table"])
print("Loaded rows:", result["records"])
print("Distinct products:", result["products"])
print("Total units:", result["total_units"])
print("First week:", result["first_week"])
print("Last week:", result["last_week"])
print("Unique product-week keys: PASS")
print("Committed database reconciliation: PASS")
print("Audit:", AUDIT_OUTPUT)
