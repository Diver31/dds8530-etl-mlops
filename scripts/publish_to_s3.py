"""Optional, explicit S3 publication of the validated weekly fact export.

No AWS upload is performed unless --execute is supplied. Requires authorized
AWS credentials and an existing private S3 bucket. Does not create public ACLs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/weekly_product_sales.csv"
AUDIT = ROOT / "reports/weekly_aggregation_audit.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-west-1")
    parser.add_argument("--prefix", default="dds8530/curated/")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not SOURCE.is_file() or not AUDIT.is_file():
        raise SystemExit("Validated ETL output and audit are required")
    audit = json.loads(AUDIT.read_text())
    assert audit["all_week_quantity_reconciliation"] == "PASS"
    with SOURCE.open("rb") as fd:
        checksum = hashlib.file_digest(fd, "sha256").hexdigest()
    key = args.prefix.strip("/") + "/weekly_product_sales_" + checksum[:16] + ".csv"
    print("S3 target:", f"s3://{args.bucket}/{key}")
    print("Verified weekly records:", audit["observed_product_week_rows"])
    print("SHA-256:", checksum)
    if not args.execute:
        print("AWS WRITE NOT EXECUTED (dry run)")
        return
    import boto3
    client = boto3.client("s3", region_name=args.region)
    with SOURCE.open("rb") as source:
        client.put_object(
            Bucket=args.bucket, Key=key, Body=source,
            ContentType="text/csv", ServerSideEncryption="AES256",
            Metadata={"source-sha256": checksum, "project": "dds8530"},
        )
    metadata = client.head_object(Bucket=args.bucket, Key=key)
    if metadata.get("Metadata", {}).get("source-sha256") != checksum:
        raise RuntimeError("S3 object metadata fingerprint mismatch")
    print("S3 WRITE AND METADATA CHECK: PASS")

if __name__ == "__main__":
    main()
