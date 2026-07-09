from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "churn_cleaned.csv"
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"

FIGURE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


BG_COLOR = "#FBF7F1"
GRID_COLOR = "#D9CEC3"
TEXT_COLOR = "#2D2727"
MODEL_COLORS = ["#8E5572", "#D95F59", "#E3A018", "#6A994E"]


def polish_bar_chart(title, ylabel):
    plt.title(title, fontsize=12, weight="bold", color=TEXT_COLOR)
    plt.xlabel("Model", fontsize=10, color=TEXT_COLOR)
    plt.ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
    plt.gca().set_facecolor(BG_COLOR)
    plt.gcf().set_facecolor(BG_COLOR)
    plt.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().spines["left"].set_color("#B8A99A")
    plt.gca().spines["bottom"].set_color("#B8A99A")


def add_score_labels(values):
    for index, value in enumerate(values):
        plt.text(index, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)


df = pd.read_csv(DATA_PATH)

print("===== Dataset loaded =====")
print(df.shape)
print(df.head())


df["Churn_binary"] = df["Churn"].map({"No": 0, "Yes": 1})

# customerID is only an identifier, so I leave it out of the model inputs.
X = df.drop(columns=["customerID", "Churn", "Churn_binary"])
y = df["Churn_binary"]


numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

print("\n===== Numeric features =====")
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

print("\n===== Train/test split =====")
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}


results = []

for model_name, model in models.items():
    print(f"\n===== Training {model_name} =====")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))


results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

print("\n===== Model comparison =====")
print(results_df)

results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)


plt.figure(figsize=(8, 4.2))
plt.bar(results_df["Model"], results_df["ROC-AUC"], color=MODEL_COLORS)
polish_bar_chart("Model Comparison by ROC-AUC", "ROC-AUC")
plt.ylim(0, 1)
plt.xticks(rotation=15)
add_score_labels(results_df["ROC-AUC"].values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_model_comparison_roc_auc.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 4.2))
plt.bar(results_df["Model"], results_df["Recall"], color=MODEL_COLORS)
polish_bar_chart("Model Comparison by Recall", "Recall")
plt.ylim(0, 1)
plt.xticks(rotation=15)
add_score_labels(results_df["Recall"].values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "08_model_comparison_recall.png", dpi=300)
plt.close()


print("\nModeling finished.")
print(f"Model comparison saved to: {REPORT_DIR / 'model_comparison.csv'}")
print(f"Figures saved to: {FIGURE_DIR}")
