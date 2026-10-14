"""Recreate the source CSV and audit from the original UCI Online Retail ZIP.

Run from the repository root on a fresh clone. This script will never overwrite
an existing staged CSV or source profile without an explicit --force flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/raw/online_retail.zip'
WORKBOOK = ROOT / 'data/raw/Online Retail.xlsx'
STAGED = ROOT / 'data/staging/online_retail.csv'
PROFILE = ROOT / 'reports/source_profile.json'
EXPECTED_ARCHIVE_SHA256 = 'f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b'
EXPECTED_COLUMNS = [
    'InvoiceNo', 'StockCode', 'Description', 'Quantity',
    'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country',
]


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_if_missing():
    if ARCHIVE.is_file():
        print('Using existing source archive:', ARCHIVE)
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ARCHIVE.with_suffix('.zip.partial')
    try:
        with requests.get(URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp.open('wb') as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        tmp.replace(ARCHIVE)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true',
                        help='Replace the existing staged CSV and source profile')
    args = parser.parse_args()
    if (STAGED.exists() or PROFILE.exists()) and not args.force:
        raise SystemExit('Staged data/profile already exists. Nothing changed. '
                         'Use --force only if you intend to regenerate them.')
    download_if_missing()
    if digest(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError('UCI source archive differs from the previously validated SHA-256; no staged output written')
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Corrupt ZIP entry: {bad}')
        if 'Online Retail.xlsx' not in archive.namelist():
            raise RuntimeError('Expected workbook missing from ZIP')
        WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
        with archive.open('Online Retail.xlsx') as source, WORKBOOK.open('wb') as target:
            shutil.copyfileobj(source, target)

    df = pd.read_excel(
        WORKBOOK,
        engine='openpyxl',
        dtype={
            'InvoiceNo': 'string', 'StockCode': 'string',
            'Description': 'string', 'CustomerID': 'string',
            'Country': 'string',
        },
    )
    if list(df.columns) != EXPECTED_COLUMNS:
        raise RuntimeError(f'Unexpected dataset columns: {list(df.columns)}')
    dates = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    invalid_dates = int((df['InvoiceDate'].notna() & dates.isna()).sum())
    if invalid_dates:
        raise RuntimeError(f'Invalid non-null invoice dates: {invalid_dates}')
    df['InvoiceDate'] = dates
    STAGED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STAGED, index=False, date_format='%Y-%m-%d %H:%M:%S')
    csv_rows = sum(len(part) for part in pd.read_csv(STAGED, chunksize=100_000))
    if csv_rows != len(df):
        raise RuntimeError('Export row count does not match original workbook')
    profile = {
        'source': 'UCI Online Retail',
        'source_url': URL,
        'source_archive_sha256': digest(ARCHIVE),
        'csv_sha256': digest(STAGED),
        'row_count': int(len(df)),
        'csv_row_count': int(csv_rows),
        'column_count': int(len(df.columns)),
        'columns': list(df.columns),
        'missing_by_column': {k: int(v) for k, v in df.isna().sum().items()},
        'exact_duplicate_rows': int(df.duplicated().sum()),
        'cancelled_invoice_rows': int(
            df['InvoiceNo'].str.startswith('C', na=False).sum()
        ),
        'negative_quantity_rows': int((df['Quantity'] < 0).sum()),
        'zero_unit_price_rows': int((df['UnitPrice'] == 0).sum()),
        'invalid_invoice_dates': invalid_dates,
        'earliest_invoice': str(dates.min()),
        'latest_invoice': str(dates.max()),
        'unique_countries': int(df['Country'].nunique()),
    }
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(profile, indent=2) + '\n')
    print('=== UCI SOURCE INGESTION ===')
    print('Rows:', profile['row_count'])
    print('Columns:', profile['column_count'])
    print('Archive SHA-256:', profile['source_archive_sha256'])
    print('CSV row reconciliation: PASS')
    print('Profile:', PROFILE.relative_to(ROOT))


if __name__ == '__main__':
    main()
