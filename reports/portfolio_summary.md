# Portfolio Summary

## Project Title

Customer Churn Prediction and Retention Strategy Analysis

## Project Type

Independent applied data analytics / business analytics portfolio project

## Tools Used

Python, pandas, matplotlib, scikit-learn, Streamlit

## Business Question

Which telecom customers are more likely to churn, and how can predicted churn risk support customer retention decisions?

## Main Methods

- Cleaned the public Telco Customer Churn dataset and fixed the `TotalCharges` data type issue.
- Explored churn patterns by contract type, payment method, internet service, monthly charges, and tenure.
- Compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Used feature importance to explain churn-related factors.
- Converted predicted churn probabilities into low-, medium-, and high-risk customer segments.
- Compared probability thresholds to show the tradeoff between precision and recall.

## Main Findings

- Month-to-month customers had a much higher churn rate than customers with longer contracts.
- Customers with shorter tenure and higher monthly charges were more likely to churn.
- Electronic check payment, fiber optic internet service, no online security, and no tech support appeared to be related to higher churn risk.
- Gradient Boosting had the highest ROC-AUC at 0.843.
- Logistic Regression had slightly higher recall at 0.559 and F1-score at 0.604, making it useful for risk scoring and explanation.

## Risk Segmentation

Predicted churn probabilities were divided into three groups:

- Low Risk: below 0.30
- Medium Risk: 0.30 to 0.60
- High Risk: above 0.60

The high-risk group had an actual churn rate of 72.1%, compared with 9.8% for the low-risk group. This suggests that the risk segmentation can help prioritize retention actions.

## Threshold Analysis

At the default threshold of 0.50, Logistic Regression reached precision of 0.657 and recall of 0.559. When the threshold was lowered to 0.30, recall increased to 0.754, but precision dropped to 0.519. This shows that churn prediction should not only be judged by accuracy. The threshold should depend on retention budget and business cost.

## Business Recommendations

1. Prioritize high-risk month-to-month customers for renewal offers.
2. Improve onboarding support for new customers with short tenure.
3. Review high monthly charge customers and offer better bundles when possible.
4. Encourage convenient automatic payment methods.
5. Provide targeted technical support to customers without support services.

## What I Learned

This project helped me understand that data analysis is not only about building models. The more important part is explaining the results and connecting them to business decisions. I also learned why recall and threshold choice matter in churn prediction.

## Resume Bullet Version

Customer Churn Prediction and Retention Strategy Analysis | Independent Data Analytics Project

- Built a customer churn analysis workflow using Python, pandas, scikit-learn, and matplotlib, covering data cleaning, exploratory analysis, model comparison, and result interpretation.
- Compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting using accuracy, precision, recall, F1-score, and ROC-AUC.
- Developed customer risk segmentation based on predicted churn probabilities and proposed targeted retention strategies.
- Added threshold analysis to explain the precision-recall tradeoff in a churn prediction context.
