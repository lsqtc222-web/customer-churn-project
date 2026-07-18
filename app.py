from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"


st.set_page_config(
    page_title="Customer Churn Portfolio",
    page_icon=":bar_chart:",
    layout="wide",
)


st.title("Customer Churn Prediction and Retention Strategy Analysis")

st.write(
    "This dashboard summarizes my customer churn portfolio project. "
    "It shows the main analysis results, model comparison, customer risk segments, "
    "and simple retention suggestions."
)


def show_image(file_name, caption):
    image_path = FIGURE_DIR / file_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {file_name}")


def load_csv(file_name):
    csv_path = REPORT_DIR / file_name
    if csv_path.exists():
        return pd.read_csv(csv_path)
    st.warning(f"Missing report file: {file_name}")
    return pd.DataFrame()


tab_overview, tab_models, tab_risk, tab_threshold, tab_actions = st.tabs(
    ["Overview", "Models", "Risk Segments", "Threshold", "Recommendations"]
)


with tab_overview:
    st.subheader("Project Overview")
    st.write(
        "The project uses the public Telco Customer Churn dataset. "
        "The goal is to understand churn patterns, compare basic machine learning models, "
        "and translate the results into retention ideas."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image("01_churn_distribution.png", "Overall churn distribution")
    with col2:
        show_image("02_churn_rate_by_contract.png", "Churn rate by contract type")

    col3, col4 = st.columns(2)
    with col3:
        show_image("05_monthly_charges_by_churn.png", "Monthly charges by churn")
    with col4:
        show_image("06_tenure_by_churn.png", "Tenure by churn")


with tab_models:
    st.subheader("Model Comparison")
    st.write(
        "I compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. "
        "Accuracy is useful, but recall is also important because missing likely churn customers "
        "may mean missing retention opportunities."
    )

    model_df = load_csv("model_comparison.csv")
    if not model_df.empty:
        st.dataframe(model_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_image("07_model_comparison_roc_auc.png", "Model comparison by ROC-AUC")
    with col2:
        show_image("08_model_comparison_recall.png", "Model comparison by recall")

    show_image("09_top_feature_importance.png", "Top churn-related features")


with tab_risk:
    st.subheader("Customer Risk Segmentation")
    st.write(
        "Predicted churn probabilities were divided into low-, medium-, and high-risk groups. "
        "This makes the model output easier to use for retention planning."
    )

    segment_df = load_csv("risk_segment_summary.csv")
    if not segment_df.empty:
        st.dataframe(segment_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        show_image("10_customer_count_by_risk_segment.png", "Customer count by risk segment")
    with col2:
        show_image("11_actual_churn_rate_by_risk_segment.png", "Actual churn rate by risk segment")

    show_image("12_average_monthly_charges_by_risk_segment.png", "Average monthly charges by risk segment")


with tab_threshold:
    st.subheader("Threshold and Recall Analysis")
    st.write(
        "Different probability thresholds change the balance between precision and recall. "
        "A lower threshold catches more churn customers, but it may also increase false positives."
    )

    threshold_df = load_csv("threshold_comparison.csv")
    if not threshold_df.empty:
        st.dataframe(threshold_df, use_container_width=True)

    show_image("13_threshold_precision_recall_tradeoff.png", "Precision and recall across thresholds")


with tab_actions:
    st.subheader("Business Recommendations")
    st.write("Based on the analysis, I would suggest these practical retention actions:")

    st.markdown(
        """
        1. Give renewal incentives to high-risk month-to-month customers.
        2. Improve onboarding support for new customers in their first few months.
        3. Review high monthly charge customers and offer better bundles where possible.
        4. Encourage stable payment methods by making them convenient.
        5. Provide targeted technical support to customers without support services.
        6. Use churn probability scores to prioritize retention campaigns.
        """
    )

    st.subheader("Limitations")
    st.write(
        "This project uses a public dataset and basic models. "
        "A real company would need newer data, customer value, retention cost, and campaign testing "
        "before using this type of model in practice."
    )
