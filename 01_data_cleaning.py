from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "churn_cleaned.csv"


df = pd.read_csv(DATA_PATH)

print("===== First five rows =====")
print(df.head())

print("\n===== Dataset shape =====")
print(df.shape)

print("\n===== Columns =====")
print(df.columns.tolist())

print("\n===== Data types =====")
print(df.dtypes)

print("\n===== Missing values before cleaning =====")
print(df.isnull().sum())

print("\n===== Duplicate rows =====")
print(df.duplicated().sum())


# TotalCharges looks numeric, but the raw file stores it as text in some rows.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\n===== Missing values after fixing TotalCharges =====")
print(df.isnull().sum())

missing_total_charges = df["TotalCharges"].isnull().sum()
median_total_charges = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

print(f"\nFilled {missing_total_charges} TotalCharges values with the median.")


df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaning finished.")
print(f"Cleaned data saved to: {OUTPUT_PATH}")
