from pathlib import Path
from datetime import date, datetime, timezone
from collections import Counter
import hashlib
import json

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RAW = Path("data/raw/holidays")
OUTPUT = Path("data/staging/uk_holidays_2010_2011.csv")
AUDIT = Path("reports/holiday_api_audit.json")

YEARS = [2010, 2011]
COUNTRY = "GB"

session = requests.Session()

retry = Retry(
    total=2,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)

records = []
sources = []

for year in YEARS:
    url = (
        "https://date.nager.at/api/v3/"
        f"PublicHolidays/{year}/{COUNTRY}"
    )

    print(f"Requesting {year} UK holidays...")

    response = session.get(url, timeout=30)
    response.raise_for_status()

    holidays = response.json()

    if not isinstance(holidays, list) or not holidays:
        raise ValueError(
            f"Missing or invalid holiday data for {year}"
        )

    raw_path = RAW / f"gb_holidays_{year}.json"
    raw_path.write_bytes(response.content)

    digest = hashlib.sha256(
        response.content
    ).hexdigest()

    for item in holidays:
        holiday_date = date.fromisoformat(item["date"])

        if holiday_date.year != year:
            raise ValueError("Holiday year mismatch.")

        if item.get("countryCode", "").upper() != COUNTRY:
            raise ValueError("Holiday country mismatch.")

        if not isinstance(item.get("global"), bool):
            raise ValueError("Invalid holiday coverage flag.")

        records.append({
            "holiday_date": holiday_date,
            "holiday_name": item["name"],
            "country_code": COUNTRY,
            "is_national": item["global"],
            "subdivisions": json.dumps(
                item.get("counties") or []
            ),
            "holiday_types": json.dumps(
                item.get("types") or []
            ),
        })

    sources.append({
        "year": year,
        "url": url,
        "record_count": len(holidays),
        "sha256": digest,
        "raw_file": str(raw_path),
    })

df = pd.DataFrame(records)

df = df.sort_values(
    ["holiday_date", "holiday_name"]
).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)

audit = {
    "source": "Nager.Date Public Holiday API v3",
    "country": COUNTRY,
    "years": YEARS,
    "retrieved_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "sources": sources,
    "holiday_records": len(df),
    "unique_holiday_dates": int(
        df["holiday_date"].nunique()
    ),
    "national_records": int(df["is_national"].sum()),
    "regional_records": int((~df["is_national"]).sum()),
    "records_by_year": {
        str(k): int(v)
        for k, v in Counter(
            d.year for d in df["holiday_date"]
        ).items()
    },
    "validation": "PASS",
    "modeling_scope":
        "UK-specific enrichment; not joined to global sales",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== HOLIDAY API EXTRACTION ===")
print("Source:", audit["source"])
print("Country:", COUNTRY)
print("Years:", YEARS)
print("Records:", audit["holiday_records"])
print("Unique dates:", audit["unique_holiday_dates"])
print("National records:", audit["national_records"])
print("Regional records:", audit["regional_records"])
print("Records by year:", audit["records_by_year"])

print("\n=== SAMPLE HOLIDAYS ===")
print(
    df[["holiday_date", "holiday_name", "is_national"]]
    .head(10)
    .to_string(index=False)
)

print("\nAPI validation: PASS")
print("CSV:", OUTPUT)
print("Audit:", AUDIT)
