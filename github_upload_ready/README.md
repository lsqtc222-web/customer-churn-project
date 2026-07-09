# Customer Churn Analysis and Retention Strategy

This is a portfolio project about customer churn in a telecom company. I built it to practise a complete data analytics workflow: cleaning data, exploring patterns, training a few machine learning models, and turning the results into simple business recommendations.

The project is written in English because I plan to use it for graduate school applications and GitHub. The style is intentionally practical rather than overly academic.

## Main Questions

I focused on four questions:

1. What kind of customers are more likely to churn?
2. Which basic machine learning model works better for this dataset?
3. Can predicted churn probability be turned into useful customer risk groups?
4. What retention actions make sense from the results?

## Dataset

The project uses the public Telco Customer Churn dataset. Each row is one customer, with information about demographics, services, account type, charges, and churn status.

Some important columns are:

| Column | Meaning |
|---|---|
| `customerID` | Customer identifier |
| `tenure` | How many months the customer has stayed |
| `Contract` | Month-to-month, one-year, or two-year contract |
| `PaymentMethod` | Payment method used by the customer |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `InternetService` | Internet service type |
| `Churn` | Whether the customer left |

The target variable is `Churn`.

## Project Structure

```text
customer-churn-project/
|-- data/
|   |-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|   `-- churn_cleaned.csv
|-- figures/
|   |-- 01_churn_distribution.png
|   |-- 02_churn_rate_by_contract.png
|   |-- 03_churn_rate_by_payment_method.png
|   |-- 04_churn_rate_by_internet_service.png
|   |-- 05_monthly_charges_by_churn.png
|   |-- 06_tenure_by_churn.png
|   |-- 07_model_comparison_roc_auc.png
|   |-- 08_model_comparison_recall.png
|   |-- 09_top_feature_importance.png
|   |-- 10_customer_count_by_risk_segment.png
|   |-- 11_actual_churn_rate_by_risk_segment.png
|   `-- 12_average_monthly_charges_by_risk_segment.png
|-- reports/
|   |-- model_comparison.csv
|   |-- feature_importance.csv
|   |-- business_insights.md
|   |-- customer_risk_scored.csv
|   |-- risk_segment_summary.csv
|   `-- customer_risk_segmentation.md
|-- 01_data_cleaning.py
|-- 02_exploratory_analysis.py
|-- 03_modeling.py
|-- 04_business_insights.py
|-- 05_customer_risk_segmentation.py
`-- README.md
```

## Workflow

### 1. Data Cleaning

The first script loads the raw dataset and checks the basic data quality. The main cleaning step is converting `TotalCharges` from text to numeric format. A small number of missing values are filled with the median.

Run:

```bash
python 01_data_cleaning.py
```

Output:

```text
data/churn_cleaned.csv
```

### 2. Exploratory Data Analysis

I used EDA to understand the churn pattern before building models. The analysis looks at churn distribution, contract type, payment method, internet service, monthly charges, and tenure.

Some example figures:

![Overall Churn Distribution](figures/01_churn_distribution.png)

![Churn Rate by Contract Type](figures/02_churn_rate_by_contract.png)

![Monthly Charges by Churn](figures/05_monthly_charges_by_churn.png)

![Tenure by Churn](figures/06_tenure_by_churn.png)

Main observations:

- Month-to-month customers have a much higher churn rate.
- Customers with shorter tenure are more likely to leave.
- Electronic check users show higher churn than other payment groups.
- Higher monthly charges seem to be linked with higher churn risk.

Run:

```bash
python 02_exploratory_analysis.py
```

### 3. Machine Learning Models

I compared four common models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The evaluation metrics are accuracy, precision, recall, F1-score, and ROC-AUC. In this case, recall is quite important because missing a likely churn customer means the company may lose a retention chance.

Run:

```bash
python 03_modeling.py
```

Outputs:

```text
reports/model_comparison.csv
figures/07_model_comparison_roc_auc.png
figures/08_model_comparison_recall.png
```

Model comparison figures:

![Model Comparison by ROC-AUC](figures/07_model_comparison_roc_auc.png)

![Model Comparison by Recall](figures/08_model_comparison_recall.png)

### 4. Feature Importance and Business Notes

For interpretation, I used the Gradient Boosting model to get feature importance. The result helps show which variables are more related to churn.

![Top Feature Importance](figures/09_top_feature_importance.png)

Important churn-related factors include:

- Contract type
- Tenure
- Monthly charges
- Payment method
- Internet service and support features

Run:

```bash
python 04_business_insights.py
```

Outputs:

```text
reports/feature_importance.csv
reports/business_insights.md
figures/09_top_feature_importance.png
```

### 5. Customer Risk Segmentation

To make the model output easier to use, I converted predicted churn probability into three risk groups:

| Segment | Rule |
|---|---|
| Low Risk | Probability below 0.30 |
| Medium Risk | Probability from 0.30 to 0.60 |
| High Risk | Probability above 0.60 |

This is a simple method, but it makes the result more useful for business actions.

![Customer Count by Risk Segment](figures/10_customer_count_by_risk_segment.png)

![Actual Churn Rate by Risk Segment](figures/11_actual_churn_rate_by_risk_segment.png)

![Average Monthly Charges by Risk Segment](figures/12_average_monthly_charges_by_risk_segment.png)

Run:

```bash
python 05_customer_risk_segmentation.py
```

Outputs:

```text
reports/customer_risk_scored.csv
reports/risk_segment_summary.csv
reports/customer_risk_segmentation.md
```

## Business Recommendations

Based on the results, I would suggest:

1. Give renewal offers to high-risk month-to-month customers.
2. Improve onboarding support for new customers in their first few months.
3. Review expensive plans and offer better bundles for high monthly charge customers.
4. Encourage stable payment methods by making them more convenient.
5. Provide more technical support to customers who do not have support services.
6. Use churn probability to rank customers before running retention campaigns.

## Limitations

This project has some clear limitations:

- The dataset is public and not live company data.
- The models are basic and only lightly tuned.
- The risk thresholds are simple assumptions for portfolio use.
- The analysis does not include customer lifetime value, campaign cost, or competitor information.
- The result should be treated as an applied analytics project, not a real production system.

Future improvements could include hyperparameter tuning, cost-sensitive modeling, customer lifetime value analysis, a dashboard, and model deployment.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts in order:

```bash
python 01_data_cleaning.py
python 02_exploratory_analysis.py
python 03_modeling.py
python 04_business_insights.py
python 05_customer_risk_segmentation.py
```

The cleaned data, figures, and reports will be saved in the `data`, `figures`, and `reports` folders.

## What I Learned

This project helped me practise both data science and business analysis. I learned that model accuracy is not the only thing that matters. For churn prediction, recall, F1-score, and the actual business use of the model are also important.

The most useful part of the project was turning model results into risk segments and retention suggestions. This made the analysis feel closer to a real business problem.
