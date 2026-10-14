from pathlib import Path
import json
import os
import re
import sys

import pandas as pd
from sqlalchemy import (
    BigInteger, Date, Text, create_engine, text
)
from sqlalchemy.engine import URL

root = Path(sys.argv[1]).resolve()
verify_only = "--verify-only" in sys.argv[2:]

if not re.fullmatch(r"[0-9a-f]{16}", root.name):
    raise ValueError("Invalid Airflow run directory.")

source = root / "data/processed/weekly_product_sales.csv"
audit = json.loads(
    (root / "reports/weekly_aggregation_audit.json").read_text()
)

df = pd.read_csv(
    source,
    dtype={"StockCode": "string", "units_sold": "int64"},
    parse_dates=["week_start"],
)

expected_rows = audit["observed_product_week_rows"]
expected_products = audit["distinct_products"]
expected_units = audit["complete_week_units"]

assert len(df) == expected_rows
assert df["StockCode"].nunique() == expected_products
assert int(df["units_sold"].sum()) == expected_units
assert not df.isna().any().any()
assert (df["units_sold"] > 0).all()
assert not df.duplicated(["StockCode", "week_start"]).any()
assert df["week_start"].dt.dayofweek.eq(0).all()

data = pd.DataFrame({
    "stock_code": df["StockCode"],
    "week_start": df["week_start"].dt.date,
    "units_sold": df["units_sold"],
})

url = URL.create(
    "postgresql+psycopg",
    username=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host=os.environ["POSTGRES_HOST"],
    port=int(os.environ["POSTGRES_PORT"]),
    database=os.environ["POSTGRES_DB"],
)

engine = create_engine(url, pool_pre_ping=True)

target = "etl.fact_weekly_sales_airflow"
stage = "airflow_stage_" + root.name

def metrics(conn, table):
    return conn.execute(text(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT stock_code) AS products,
            COALESCE(SUM(units_sold), 0) AS units,
            MIN(week_start) AS first_week,
            MAX(week_start) AS last_week
        FROM {table}
    """)).mappings().one()

def validate(result):
    assert int(result["rows"]) == expected_rows
    assert int(result["products"]) == expected_products
    assert int(result["units"]) == expected_units
    assert str(result["first_week"]) == audit["first_complete_week"]
    assert str(result["last_week"]) == audit["last_complete_week"]

if not verify_only:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS etl"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS
            etl.fact_weekly_sales_airflow (
                stock_code TEXT NOT NULL,
                week_start DATE NOT NULL,
                units_sold BIGINT NOT NULL
                    CHECK (units_sold > 0),
                loaded_at TIMESTAMPTZ NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_code, week_start)
            )
        """))

        data.to_sql(
            stage,
            con=conn,
            schema="etl",
            if_exists="fail",
            index=False,
            chunksize=5000,
            dtype={
                "stock_code": Text(),
                "week_start": Date(),
                "units_sold": BigInteger(),
            },
        )

        staged = metrics(conn, f"etl.{stage}")
        validate(staged)

        conn.execute(text(f"DELETE FROM {target}"))

        conn.execute(text(f"""
            INSERT INTO {target}
                (stock_code, week_start, units_sold)
            SELECT stock_code, week_start, units_sold
            FROM etl.{stage}
        """))

        validate(metrics(conn, target))
        conn.execute(text(f"DROP TABLE etl.{stage}"))

# Verify the committed database state independently.
with engine.connect() as conn:
    committed = metrics(conn, target)
    validate(committed)

engine.dispose()

result = {
    "status": "PASS",
    "mode": "verify" if verify_only else "load",
    "table": target,
    "records": int(committed["rows"]),
    "products": int(committed["products"]),
    "units": int(committed["units"]),
    "first_week": str(committed["first_week"]),
    "last_week": str(committed["last_week"]),
}

if not verify_only:
    output = root / "reports/airflow_database_audit.json"
    output.write_text(json.dumps(result, indent=2))

print("=== AIRFLOW POSTGRESQL VALIDATION ===")
print("Mode:", result["mode"])
print("Table:", target)
print("Records:", result["records"])
print("Products:", result["products"])
print("Units:", result["units"])
print("Committed database reconciliation: PASS")
