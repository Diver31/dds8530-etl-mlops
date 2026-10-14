from pathlib import Path
import pandas as pd

source = Path("data/processed/retail_sales_candidates.csv")
output = Path("reports/stock_code_review.csv")

df = pd.read_csv(
    source,
    dtype={
        "StockCode": "string",
        "Description": "string",
    },
    usecols=[
        "StockCode",
        "Description",
        "Quantity",
        "line_value",
    ],
)

df["StockCode"] = df["StockCode"].str.strip().str.upper()
df["Description"] = df["Description"].fillna(
    "[MISSING]"
).str.strip()

# Allow five digits followed by up to three letters.
# This is a screening rule, not proof of merchandise status.
standard = df["StockCode"].str.fullmatch(
    r"\d{5}[A-Z]{0,3}", na=False
)

review = df.loc[~standard].copy()

rows = []

for code, group in review.groupby("StockCode"):
    descriptions = (
        group["Description"]
        .value_counts()
        .head(2)
        .index
        .tolist()
    )

    rows.append({
        "StockCode": code,
        "Records": len(group),
        "TotalQuantity": int(group["Quantity"].sum()),
        "TotalValue": round(group["line_value"].sum(), 2),
        "MainDescription": descriptions[0] if descriptions else "",
        "OtherDescription": descriptions[1] if len(descriptions) > 1 else "",
    })

result = pd.DataFrame(rows).sort_values(
    "Records", ascending=False
)

result.to_csv(output, index=False)

print("\n=== STOCK CODE REVIEW ===")
print("Candidate sales:", len(df))
print("Rows requiring code review:", len(review))
print("Distinct codes requiring review:", len(result))

print("\n=== TOP 20 CODES ===")

for _, row in result.head(20).iterrows():
    print(
        f"{row['StockCode']:<18} "
        f"Rows: {row['Records']:>5}  "
        f"Qty: {row['TotalQuantity']:>8}  "
        f"Value: {row['TotalValue']:>12,.2f}  "
        f"Description: {row['MainDescription'][:55]}"
    )

print("\n=== IMPORTANT CODES ===")

important = [
    "POST",
    "DOT",
    "M",
    "C2",
    "BANK CHARGES",
    "GIFT_0001_20",
    "DCGSSGIRL",
]

for code in important:
    match = result[result["StockCode"] == code]
    if not match.empty:
        row = match.iloc[0]
        print(
            f"{code}: {row['Records']} rows | "
            f"{row['MainDescription']}"
        )

print("\nAudit saved to:", output)
print("Stock-code review: COMPLETE")
