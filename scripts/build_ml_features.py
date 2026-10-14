from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA = Path("data/processed")
REPORTS = Path("reports")
OUTPUT = DATA / "uk_model_ready.csv"
AUDIT = REPORTS / "ml_feature_audit.json"

source_audit = json.loads(
    (REPORTS / "uk_weekly_audit.json").read_text()
)

weekly = pd.read_csv(
    DATA / "uk_weekly_sales_observed.csv",
    dtype={"StockCode": "string", "units_sold": "int64"},
    parse_dates=["week_start"],
)

if len(weekly) != source_audit["uk_product_week_rows"]:
    raise ValueError("Source row count mismatch.")
if int(weekly["units_sold"].sum()) != source_audit["uk_units_complete_weeks"]:
    raise ValueError("Source quantity mismatch.")
if weekly[["StockCode", "week_start", "units_sold"]].isna().any().any():
    raise ValueError("Missing required source values.")
if (weekly["units_sold"] <= 0).any():
    raise ValueError("Unexpected nonpositive observed sales.")
if weekly.duplicated(["StockCode", "week_start"]).any():
    raise ValueError("Duplicate product-week keys.")

# Establish 52 chronological reporting weeks.
weeks = pd.date_range(
    start=source_audit["first_complete_week"],
    end=source_audit["last_complete_week"],
    freq="W-MON",
)

if len(weeks) != 52:
    raise ValueError("Expected exactly 52 weeks.")

train_weeks = weeks[:40]
test_weeks = weeks[40:]
test_start = test_weeks[0]

# Select products using development-period activity only.
history = weekly.loc[
    weekly["week_start"].isin(train_weeks)
]

coverage = history.groupby("StockCode")[
    "week_start"
].nunique()

eligible = sorted(
    coverage.loc[coverage >= 20].index.tolist()
)

if not eligible:
    raise ValueError("No products meet the history requirement.")

selected = weekly.loc[
    weekly["StockCode"].isin(eligible)
].copy()

# Do not generate zeros before a product's first
# observed transaction in the development period.
first_seen = history.groupby("StockCode")[
    "week_start"
].min()

grid = pd.MultiIndex.from_product(
    [eligible, weeks],
    names=["StockCode", "week_start"],
).to_frame(index=False)

grid["first_seen"] = grid["StockCode"].map(first_seen)
grid = grid.loc[
    grid["week_start"] >= grid["first_seen"]
].drop(columns="first_seen")

panel = grid.merge(
    selected[["StockCode", "week_start", "units_sold"]],
    on=["StockCode", "week_start"],
    how="left",
    validate="one_to_one",
    indicator=True,
)

observed_count = int(panel["_merge"].eq("both").sum())
zero_filled = int(panel["_merge"].eq("left_only").sum())

if observed_count != len(selected):
    raise RuntimeError("Observed product-week reconciliation failed.")

panel["units_sold"] = (
    panel["units_sold"].fillna(0).astype("int64")
)
panel = panel.drop(columns="_merge")

panel = panel.sort_values(
    ["StockCode", "week_start"]
).reset_index(drop=True)

# Integrate nationwide UK holidays.
holidays = pd.read_csv(
    "data/staging/uk_holidays_2010_2011.csv",
    dtype={"is_national": "string"},
    parse_dates=["holiday_date"],
)

flags = holidays["is_national"].str.lower()

if flags.isna().any() or not flags.isin(
    ["true", "false"]
).all():
    raise ValueError("Invalid holiday coverage flags.")

national = holidays.loc[flags.eq("true")].copy()

national_weeks = (
    national["holiday_date"]
    - pd.to_timedelta(
        national["holiday_date"].dt.dayofweek,
        unit="D",
    )
).drop_duplicates()

panel["uk_holiday_week"] = (
    panel["week_start"].isin(national_weeks)
    .astype("int8")
)

# Calendar features are known before the forecast week.
week_number = (
    panel["week_start"].dt.isocalendar().week
    .astype("int16")
)

panel["week_sin"] = np.sin(
    2 * np.pi * week_number / 52.1775
)
panel["week_cos"] = np.cos(
    2 * np.pi * week_number / 52.1775
)

# Shift historical observations to prevent target leakage.
group = panel.groupby("StockCode", sort=False)["units_sold"]

for lag in (1, 2, 4):
    panel[f"lag_{lag}"] = group.shift(lag)

panel["rolling_prev_4"] = group.transform(
    lambda series: series.shift(1).rolling(
        window=4, min_periods=4
    ).mean()
)

# First four product weeks lack sufficient lag history.
features = ["lag_1", "lag_2", "lag_4", "rolling_prev_4"]
model = panel.dropna(subset=features).copy()

model["split"] = np.where(
    model["week_start"] < test_start,
    "train",
    "test",
)

train = model.loc[model["split"] == "train"]
test = model.loc[model["split"] == "test"]

if train.empty or test.empty:
    raise RuntimeError("Training or test dataset is empty.")

if train["week_start"].max() >= test["week_start"].min():
    raise RuntimeError("Chronological split failed.")

if model[features].isna().any().any():
    raise RuntimeError("Missing model features.")

if model.duplicated(["StockCode", "week_start"]).any():
    raise RuntimeError("Duplicate modeling keys.")

model.to_csv(
    OUTPUT, index=False, date_format="%Y-%m-%d"
)

audit = {
    "source_product_week_rows": len(weekly),
    "eligibility_rule": "At least 20 observed weeks in first 40 weeks",
    "eligible_products": len(eligible),
    "observed_selected_product_weeks": observed_count,
    "zero_filled_product_weeks": zero_filled,
    "model_ready_rows": len(model),
    "train_rows": len(train),
    "test_rows": len(test),
    "train_last_week": str(train_weeks[-1].date()),
    "test_first_week": str(test_weeks[0].date()),
    "test_last_week": str(test_weeks[-1].date()),
    "national_holiday_weeks_train": int(
        national_weeks.isin(train_weeks).sum()
    ),
    "national_holiday_weeks_test": int(
        national_weeks.isin(test_weeks).sum()
    ),
    "evaluation": "Sequential one-week-ahead forecasting",
    "missing_week_assumption":
        "Zero recorded sales after first observed sale; inventory unknown",
    "normalization": "Deferred to training-only sklearn pipeline",
    "validation": "PASS",
}

AUDIT.write_text(json.dumps(audit, indent=2))

print("\n=== ML FEATURE ENGINEERING ===")
print("Source product-week records:", len(weekly))
print("Products eligible from training history:", len(eligible))
print("Observed selected product-weeks:", observed_count)
print("Zero-filled product-weeks:", zero_filled)
print("Model-ready rows:", len(model))
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Training last week:", train_weeks[-1].date())
print("Test first week:", test_weeks[0].date())
print("Test last week:", test_weeks[-1].date())
print("National holiday weeks in train:",
      audit["national_holiday_weeks_train"])
print("National holiday weeks in test:",
      audit["national_holiday_weeks_test"])
print("Chronological validation: PASS")
print("Feature validation: PASS")
print("Output:", OUTPUT)
print("Audit:", AUDIT)
