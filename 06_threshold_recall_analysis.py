from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
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
PRECISION_COLOR = "#6A994E"
RECALL_COLOR = "#D95F59"


df = pd.read_csv(DATA_PATH)
df["Churn_binary"] = df["Churn"].map({"No": 0, "Yes": 1})

# This uses the same feature setup and preprocessing style as 03_modeling.py.
X = df.drop(columns=["customerID", "Churn", "Churn_binary"])
y = df["Churn_binary"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

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

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
results = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    results.append(
        {
            "Threshold": threshold,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Predicted_Churn_Customers": int(y_pred.sum()),
        }
    )

threshold_df = pd.DataFrame(results)
threshold_path = REPORT_DIR / "threshold_comparison.csv"
threshold_df.to_csv(threshold_path, index=False)

print("===== Threshold comparison =====")
print(threshold_df)


plt.figure(figsize=(7.5, 4.5))
plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    linewidth=2,
    color=PRECISION_COLOR,
    label="Precision",
)
plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    linewidth=2,
    color=RECALL_COLOR,
    label="Recall",
)
plt.title("Precision and Recall Across Churn Probability Thresholds", fontsize=12, weight="bold", color=TEXT_COLOR)
plt.xlabel("Probability Threshold", fontsize=10, color=TEXT_COLOR)
plt.ylabel("Score", fontsize=10, color=TEXT_COLOR)
plt.ylim(0, 1)
plt.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
plt.gca().set_facecolor(BG_COLOR)
plt.gcf().set_facecolor(BG_COLOR)
plt.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines["left"].set_color("#B8A99A")
plt.gca().spines["bottom"].set_color("#B8A99A")
plt.legend(frameon=False)
plt.tight_layout()
figure_path = FIGURE_DIR / "13_threshold_precision_recall_tradeoff.png"
plt.savefig(figure_path, dpi=300)
plt.close()


best_recall_row = threshold_df.sort_values("Recall", ascending=False).iloc[0]
default_row = threshold_df[threshold_df["Threshold"] == 0.50].iloc[0]

analysis_text = f"""# Threshold and Recall Analysis

## Purpose

This script checks how the churn prediction result changes when the probability threshold is adjusted. The model is the Logistic Regression pipeline used in the main modeling workflow. I used Logistic Regression here because it is easier to explain and it had relatively good recall in the model comparison.

## Results

The threshold comparison is saved in `reports/threshold_comparison.csv`.

At the default threshold of 0.50, the model reached precision of {default_row["Precision"]:.3f}, recall of {default_row["Recall"]:.3f}, F1-score of {default_row["F1-score"]:.3f}, and accuracy of {default_row["Accuracy"]:.3f}. It predicted {int(default_row["Predicted_Churn_Customers"])} customers in the test set as churn.

The lowest tested threshold, {best_recall_row["Threshold"]:.2f}, gave the highest recall of {best_recall_row["Recall"]:.3f}. This means the model caught more actual churn customers, but it also flagged more customers as high risk. In a real retention campaign, this could increase false positives and campaign cost.

## Business Meaning

In churn prediction, recall matters because missing a high-risk customer may mean losing a chance to contact the customer before they leave. However, precision also matters because retention offers, service calls, and discounts all cost money.

A lower threshold may be useful when the company has enough retention budget and wants to reach more possible churn customers. A higher threshold may be better when the company wants to contact fewer customers with stronger churn signals.

There is no single threshold that is always best. The final choice should depend on business cost, customer value, and the size of the retention budget.
"""

analysis_path = REPORT_DIR / "threshold_recall_analysis.md"
with open(analysis_path, "w", encoding="utf-8") as f:
    f.write(analysis_text)

print(f"\nThreshold comparison saved to: {threshold_path}")
print(f"Threshold figure saved to: {figure_path}")
print(f"Threshold explanation saved to: {analysis_path}")
