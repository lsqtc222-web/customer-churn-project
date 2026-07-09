from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "churn_cleaned.csv"
FIGURE_DIR = PROJECT_DIR / "figures"

FIGURE_DIR.mkdir(exist_ok=True)


BG_COLOR = "#FBF7F1"
GRID_COLOR = "#D9CEC3"
TEXT_COLOR = "#2D2727"
PALETTE = {
    "coral": "#D95F59",
    "mustard": "#E3A018",
    "green": "#6A994E",
    "purple": "#8E5572",
    "rose": "#C65D7B",
    "brown": "#8A5A44",
}


def polish_plot(title, xlabel, ylabel):
    plt.title(title, fontsize=12, weight="bold", color=TEXT_COLOR)
    plt.xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    plt.ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.55, color=GRID_COLOR)
    plt.gca().set_facecolor(BG_COLOR)
    plt.gcf().set_facecolor(BG_COLOR)
    plt.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)
    plt.gca().spines["left"].set_color("#B8A99A")
    plt.gca().spines["bottom"].set_color("#B8A99A")


def add_bar_labels(values, fmt="{:.1%}"):
    for index, value in enumerate(values):
        plt.text(index, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


df = pd.read_csv(DATA_PATH)

print("===== Dataset loaded =====")
print(df.head())
print(df.shape)

print("\n===== Churn distribution =====")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))


churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])

plt.figure(figsize=(6, 4))
plt.bar(churn_counts.index, churn_counts.values, color=[PALETTE["green"], PALETTE["coral"]])
polish_plot("Overall Churn Distribution", "Churn", "Number of Customers")
for i, value in enumerate(churn_counts.values):
    plt.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_churn_distribution.png", dpi=300)
plt.close()


contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda rows: (rows == "Yes").mean()
).sort_values(ascending=False)

print("\n===== Churn rate by contract =====")
print(contract_churn)

plt.figure(figsize=(7, 4))
plt.bar(contract_churn.index, contract_churn.values, color=PALETTE["coral"])
polish_plot("Churn Rate by Contract Type", "Contract Type", "Churn Rate")
plt.ylim(0, max(contract_churn.values) + 0.08)
plt.xticks(rotation=15)
add_bar_labels(contract_churn.values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_churn_rate_by_contract.png", dpi=300)
plt.close()


payment_churn = df.groupby("PaymentMethod")["Churn"].apply(
    lambda rows: (rows == "Yes").mean()
).sort_values(ascending=False)

print("\n===== Churn rate by payment method =====")
print(payment_churn)

plt.figure(figsize=(9, 4.6))
payment_colors = [PALETTE["rose"], PALETTE["mustard"], PALETTE["purple"], PALETTE["green"]]
plt.bar(payment_churn.index, payment_churn.values, color=payment_colors)
polish_plot("Churn Rate by Payment Method", "Payment Method", "Churn Rate")
plt.ylim(0, max(payment_churn.values) + 0.08)
plt.xticks(rotation=25, ha="right")
add_bar_labels(payment_churn.values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_churn_rate_by_payment_method.png", dpi=300)
plt.close()


internet_churn = df.groupby("InternetService")["Churn"].apply(
    lambda rows: (rows == "Yes").mean()
).sort_values(ascending=False)

print("\n===== Churn rate by internet service =====")
print(internet_churn)

plt.figure(figsize=(7, 4))
plt.bar(
    internet_churn.index,
    internet_churn.values,
    color=[PALETTE["purple"], PALETTE["mustard"], PALETTE["green"]],
)
polish_plot("Churn Rate by Internet Service", "Internet Service", "Churn Rate")
plt.ylim(0, max(internet_churn.values) + 0.08)
add_bar_labels(internet_churn.values)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_churn_rate_by_internet_service.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 4.2))
plt.hist(
    df[df["Churn"] == "No"]["MonthlyCharges"],
    bins=30,
    alpha=0.72,
    label="No Churn",
    color=PALETTE["green"],
)
plt.hist(
    df[df["Churn"] == "Yes"]["MonthlyCharges"],
    bins=30,
    alpha=0.72,
    label="Churn",
    color=PALETTE["coral"],
)
polish_plot("Monthly Charges Distribution by Churn", "Monthly Charges", "Number of Customers")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_monthly_charges_by_churn.png", dpi=300)
plt.close()


plt.figure(figsize=(7, 4.2))
plt.hist(
    df[df["Churn"] == "No"]["tenure"],
    bins=30,
    alpha=0.72,
    label="No Churn",
    color=PALETTE["green"],
)
plt.hist(
    df[df["Churn"] == "Yes"]["tenure"],
    bins=30,
    alpha=0.72,
    label="Churn",
    color=PALETTE["coral"],
)
polish_plot("Tenure Distribution by Churn", "Tenure in Months", "Number of Customers")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_tenure_by_churn.png", dpi=300)
plt.close()


print("\nEDA finished.")
print(f"Figures saved to: {FIGURE_DIR}")
