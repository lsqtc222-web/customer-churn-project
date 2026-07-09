from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
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
BAR_COLOR = "#8E5572"


df = pd.read_csv(DATA_PATH)
df["Churn_binary"] = df["Churn"].map({"No": 0, "Yes": 1})

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


# I use Gradient Boosting here because it gave the strongest ROC-AUC in this project.
model = GradientBoostingClassifier(random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

pipeline.fit(X_train, y_train)


onehot = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_cat_features = onehot.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + encoded_cat_features.tolist()

feature_importance = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": all_feature_names,
        "Importance": feature_importance,
    }
).sort_values(by="Importance", ascending=False)

importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

print("===== Top 15 important features =====")
print(importance_df.head(15))


top_features = importance_df.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color=BAR_COLOR)
plt.title("Top 15 Features Related to Customer Churn", fontsize=12, weight="bold", color=TEXT_COLOR)
plt.xlabel("Feature Importance", fontsize=10, color=TEXT_COLOR)
plt.ylabel("Feature", fontsize=10, color=TEXT_COLOR)
plt.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
plt.gca().set_facecolor(BG_COLOR)
plt.gcf().set_facecolor(BG_COLOR)
plt.tick_params(colors=TEXT_COLOR, labelsize=9)
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)
plt.gca().spines["left"].set_color("#B8A99A")
plt.gca().spines["bottom"].set_color("#B8A99A")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "09_top_feature_importance.png", dpi=300)
plt.close()


insights = """# Customer Churn Analysis: Business Notes

## Project Aim

I used the Telco Customer Churn dataset to practise a full data analysis workflow. The main question is simple: which customers look more likely to leave, and what can a company do before that happens?

This project is not a production churn system. It is a portfolio project that connects data cleaning, exploratory analysis, machine learning, and business thinking.

## Model Summary

I compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Gradient Boosting had the best ROC-AUC in my test results, while Logistic Regression gave slightly better recall and F1-score.

For churn prediction, recall is important because a company would rather notice more risky customers than miss them completely. At the same time, precision still matters because retention campaigns cost money.

## Main Churn Patterns

The analysis points to a few customer groups that may need more attention:

- Customers on month-to-month contracts
- Customers with short tenure
- Customers with higher monthly charges
- Customers paying by electronic check
- Customers without technical support or similar service protection

These patterns make business sense. A customer with a flexible contract and little service support has fewer reasons to stay if the price feels high.

## Suggested Retention Ideas

1. Offer small renewal benefits for month-to-month customers who show high churn probability.
2. Give new customers more support in the first few months, because early churn is easier to miss.
3. Review high monthly charge customers and offer better bundles where possible.
4. Encourage more stable payment methods, but do it through convenience rather than pressure.
5. Use technical support as a retention tool, especially for customers who already have internet service.

## How the Model Could Be Used

The model can give each customer a churn probability. I would not use the score as an automatic decision. A better use is to rank customers and help the retention team decide who should be contacted first.

## Limitations

The dataset is public and does not include live customer behaviour, complaint history, competitor prices, or retention campaign results. The models are also basic and only lightly tuned. In a real company, I would test the model with more recent data and compare the cost of retention actions with customer value.

## What I Learned

This project helped me understand that a model is only one part of a business analysis. The more useful part is turning model outputs into clear customer groups and practical actions.
"""

with open(REPORT_DIR / "business_insights.md", "w", encoding="utf-8") as f:
    f.write(insights)

print("\nBusiness insights saved to reports/business_insights.md")
print("Feature importance saved to reports/feature_importance.csv")
print("Feature importance figure saved to figures/09_top_feature_importance.png")
