from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "churn_cleaned.csv"
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

FIGURE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


BG_COLOR = "#FBF7F1"
GRID_COLOR = "#D9CEC3"
TEXT_COLOR = "#2D2727"
SEGMENT_COLORS = {
    "Low Risk": "#6A994E",
    "Medium Risk": "#E3A018",
    "High Risk": "#D95F59",
}


def polish_chart(title, ylabel):
    plt.title(title, fontsize=12, weight="bold", color=TEXT_COLOR)
    plt.xlabel("Risk Segment", fontsize=10, color=TEXT_COLOR)
    plt.ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
    plt.gca().set_facecolor(BG_COLOR)
    plt.gcf().set_facecolor(BG_COLOR)
    plt.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().spines["left"].set_color("#B8A99A")
    plt.gca().spines["bottom"].set_color("#B8A99A")


def add_labels(values, kind="number"):
    for index, value in enumerate(values):
        if kind == "percent":
            label = f"{value:.1%}"
        elif kind == "money":
            label = f"${value:.1f}"
        else:
            label = f"{int(value):,}"
        plt.text(index, value, label, ha="center", va="bottom", fontsize=9)


df = pd.read_csv(DATA_PATH)
df["Churn_binary"] = df["Churn"].map({"No": 0, "Yes": 1})

X = df.drop(columns=["customerID", "Churn", "Churn_binary"])
y = df["Churn_binary"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

print("===== Numeric features =====")
print(numeric_features)

print("\n===== Categorical features =====")
print(categorical_features)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Logistic Regression is easy to explain and performed well on recall in the model comparison.
model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)


df["Predicted_Churn_Probability"] = pipeline.predict_proba(X)[:, 1]

df["Risk_Segment"] = pd.cut(
    df["Predicted_Churn_Probability"],
    bins=[0, 0.30, 0.60, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    include_lowest=True,
)


scored_data_path = REPORT_DIR / "customer_risk_scored.csv"
df.to_csv(scored_data_path, index=False)


segment_summary = df.groupby("Risk_Segment", observed=False).agg(
    Customer_Count=("customerID", "count"),
    Average_Predicted_Churn_Probability=("Predicted_Churn_Probability", "mean"),
    Actual_Churn_Rate=("Churn_binary", "mean"),
    Average_Tenure=("tenure", "mean"),
    Average_MonthlyCharges=("MonthlyCharges", "mean"),
).reset_index()

segment_summary_path = REPORT_DIR / "risk_segment_summary.csv"
segment_summary.to_csv(segment_summary_path, index=False)

print("\n===== Risk segment summary =====")
print(segment_summary)


ordered_segments = ["Low Risk", "Medium Risk", "High Risk"]
plot_colors = [SEGMENT_COLORS[segment] for segment in ordered_segments]
segment_summary["Risk_Segment"] = segment_summary["Risk_Segment"].astype(str)
segment_summary = segment_summary.set_index("Risk_Segment").loc[ordered_segments].reset_index()


plt.figure(figsize=(7, 4))
plt.bar(segment_summary["Risk_Segment"], segment_summary["Customer_Count"], color=plot_colors)
polish_chart("Customer Count by Risk Segment", "Number of Customers")
add_labels(segment_summary["Customer_Count"].values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "10_customer_count_by_risk_segment.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 4))
plt.bar(segment_summary["Risk_Segment"], segment_summary["Actual_Churn_Rate"], color=plot_colors)
polish_chart("Actual Churn Rate by Risk Segment", "Actual Churn Rate")
plt.ylim(0, max(segment_summary["Actual_Churn_Rate"]) + 0.10)
add_labels(segment_summary["Actual_Churn_Rate"].values, kind="percent")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "11_actual_churn_rate_by_risk_segment.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 4))
plt.bar(segment_summary["Risk_Segment"], segment_summary["Average_MonthlyCharges"], color=plot_colors)
polish_chart("Average Monthly Charges by Risk Segment", "Average Monthly Charges")
plt.ylim(0, max(segment_summary["Average_MonthlyCharges"]) + 12)
add_labels(segment_summary["Average_MonthlyCharges"].values, kind="money")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "12_average_monthly_charges_by_risk_segment.png", dpi=300)
plt.close()


insights = """# Customer Risk Segmentation

## Purpose

After building a churn model, I used the predicted churn probability to split customers into three groups. This makes the model easier to use for business decisions.

- Low Risk: predicted churn probability below 0.30
- Medium Risk: predicted churn probability from 0.30 to 0.60
- High Risk: predicted churn probability above 0.60

## Results

The customer-level scored data is saved in `reports/customer_risk_scored.csv`.

The summary by risk segment is saved in `reports/risk_segment_summary.csv`.

In this result, the high-risk group has a much higher actual churn rate and a shorter average tenure. This is a useful check because the groups are not only different in model score; they also show different real churn behaviour.

## Business Meaning

The segmentation can help a company decide where to spend retention effort. Low-risk customers may only need normal service quality. Medium-risk customers can be monitored. High-risk customers should be contacted earlier, especially when they also have high monthly charges or short tenure.

## Suggested Actions

1. Low Risk: keep regular service quality and avoid unnecessary discounts.
2. Medium Risk: watch for changes in usage or payment behaviour.
3. High Risk: use targeted calls, better onboarding, renewal incentives, or service support.

## Limitations

The 0.30 and 0.60 thresholds are simple choices for this portfolio project. In a real company, I would adjust them by customer lifetime value, retention cost, and campaign budget.
"""

with open(REPORT_DIR / "customer_risk_segmentation.md", "w", encoding="utf-8") as f:
    f.write(insights)


print("\nCustomer risk segmentation finished.")
print(f"Scored customer data saved to: {scored_data_path}")
print(f"Risk segment summary saved to: {segment_summary_path}")
print("Figures saved:")
print(FIGURE_DIR / "10_customer_count_by_risk_segment.png")
print(FIGURE_DIR / "11_actual_churn_rate_by_risk_segment.png")
print(FIGURE_DIR / "12_average_monthly_charges_by_risk_segment.png")
print(REPORT_DIR / "customer_risk_segmentation.md")
