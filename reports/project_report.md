# Customer Churn Prediction and Retention Strategy Analysis

## 1. Introduction

This project is an applied data analytics project about customer churn in a telecom company. I used the public Telco Customer Churn dataset to practise a practical workflow, from data cleaning to business interpretation.

The goal of this project is not to create a new research method. It is also not a production system. The goal is to show that I can use Python and basic machine learning methods to understand a business problem, compare model results, and explain what the results may mean for customer retention.

The project includes six main parts: data cleaning, exploratory data analysis, model comparison, feature importance, customer risk segmentation, and threshold analysis. The final output includes figures, CSV reports, markdown reports, and a simple Streamlit dashboard.

## 2. Business Problem

Customer churn means that a customer stops using a company service. For subscription businesses, churn is important because losing existing customers can reduce future revenue. It may also cost more to attract new customers than to keep current customers.

In this project, I focused on three business questions:

1. Which customer features are related to churn?
2. Which basic machine learning models perform better for churn prediction?
3. How can model outputs be used to support retention decisions?

The business use of the model is not to automatically decide what happens to a customer. A better use is to help a retention team rank customers by risk and choose where to spend attention first.

## 3. Dataset Description

The dataset is the public Telco Customer Churn dataset. Each row represents one customer. The dataset includes customer demographic information, account information, services used, contract type, payment method, monthly charges, total charges, and churn status.

The target variable is `Churn`, with two values: `Yes` and `No`. In the modeling scripts, I converted it into a binary variable where churn is 1 and non-churn is 0.

Some important variables are `tenure`, `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `OnlineSecurity`, and `TechSupport`. These variables are useful because they describe both customer relationship length and service experience.

## 4. Data Cleaning

The cleaning process is in `01_data_cleaning.py`. I first loaded the raw CSV file and checked the dataset shape, column names, data types, missing values, and duplicate rows.

The main cleaning issue was `TotalCharges`. Although it should be numeric, the raw file stores it as text in some rows. I converted `TotalCharges` to numeric format. After conversion, 11 values became missing, so I filled them with the median value. This is a simple approach, but it is acceptable here because the number of missing values is small compared with the full dataset.

The cleaned file is saved as `data/churn_cleaned.csv`.

## 5. Exploratory Data Analysis

The exploratory analysis is in `02_exploratory_analysis.py`. I used it to understand the dataset before modeling.

The churn distribution figure shows that more customers stayed than left, but the churn group is still large enough to be important. This also means accuracy alone can be misleading, because a model could look acceptable by mostly predicting the larger non-churn group.

The contract analysis shows a clear churn pattern. Customers with month-to-month contracts have a much higher churn rate than customers with one-year or two-year contracts. This makes sense because month-to-month customers can leave more easily.

Payment method also shows differences. Electronic check users have a higher churn rate than customers using automatic bank transfer or credit card payment. This may reflect payment convenience, customer group differences, or other service experience factors.

The tenure distribution suggests that customers who churn often have shorter tenure. This means early customer experience may be important. The monthly charges figure also suggests that customers who churn tend to have higher monthly charges.

## 6. Modeling Methodology

The modeling process is in `03_modeling.py`. I removed `customerID` because it is only an identifier and should not be used as a predictive feature.

I separated numeric features and categorical features. Numeric features were standardized with `StandardScaler`. Categorical features were transformed using one-hot encoding. I used a train-test split with stratification so that the churn ratio stayed similar in both sets.

I compared four common classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

These models are not highly tuned. I kept them simple because the project is meant to be readable and suitable for a student portfolio.

## 7. Model Evaluation

The model comparison results are saved in `reports/model_comparison.csv`. I used accuracy, precision, recall, F1-score, and ROC-AUC.

Gradient Boosting achieved the highest ROC-AUC at 0.843. Its accuracy was 0.806, precision was 0.674, recall was 0.524, and F1-score was 0.589.

Logistic Regression also performed well. Its ROC-AUC was 0.842, very close to Gradient Boosting. It had accuracy of 0.806, precision of 0.657, recall of 0.559, and F1-score of 0.604. Logistic Regression had slightly better recall and F1-score in this result.

For churn prediction, recall is important because missing a customer who is likely to churn may mean missing a chance to keep that customer. Precision is also important because retention actions cost money. This is why I did not only look at accuracy.

Decision Tree had the weakest ROC-AUC at 0.657. Random Forest performed better than Decision Tree but was behind Gradient Boosting and Logistic Regression in this result.

## 8. Feature Importance and Churn Drivers

The feature importance analysis is in `04_business_insights.py`, using Gradient Boosting. The output is saved in `reports/feature_importance.csv`, and the figure is saved as `figures/09_top_feature_importance.png`.

The most important feature was `Contract_Month-to-month`. Other important features included `tenure`, `InternetService_Fiber optic`, `TotalCharges`, `MonthlyCharges`, `OnlineSecurity_No`, `PaymentMethod_Electronic check`, and `TechSupport_No`.

These results are reasonable from a business point of view. Customers on flexible contracts can leave more easily. Customers with short tenure may not have built a strong relationship with the company. Customers with higher monthly charges may be more sensitive to service quality or price. Customers without online security or tech support may have weaker service protection.

This part of the project helped me understand that model interpretation is important. A model score is useful, but decision makers also need to know what factors may be driving the risk.

## 9. Customer Risk Segmentation

The customer risk segmentation is in `05_customer_risk_segmentation.py`. I used Logistic Regression to produce predicted churn probabilities for all customers. Then I divided customers into three groups:

- Low Risk: predicted churn probability below 0.30
- Medium Risk: predicted churn probability from 0.30 to 0.60
- High Risk: predicted churn probability above 0.60

The result is saved in `reports/customer_risk_scored.csv` and `reports/risk_segment_summary.csv`.

The low-risk group has 4,374 customers and an actual churn rate of 9.8%. The medium-risk group has 1,630 customers and an actual churn rate of 42.3%. The high-risk group has 1,039 customers and an actual churn rate of 72.1%.

The average tenure also changes across groups. Low-risk customers have average tenure of 42.4 months, medium-risk customers have 20.4 months, and high-risk customers have 8.9 months. This supports the earlier observation that short tenure is related to churn risk.

This segmentation makes the model easier to use. Instead of only saying whether a customer will churn, the company can rank customers and plan different actions for each risk group.

## 10. Threshold and Recall Analysis

The threshold analysis is in `06_threshold_recall_analysis.py`. I added this part because the default threshold of 0.50 is not always suitable for churn prediction.

I compared thresholds of 0.30, 0.40, 0.50, 0.60, and 0.70 using Logistic Regression on the test set. The result is saved in `reports/threshold_comparison.csv`.

At threshold 0.50, precision was 0.657, recall was 0.559, F1-score was 0.604, and accuracy was 0.806. The model predicted 318 customers in the test set as churn.

At threshold 0.30, recall increased to 0.754, and the model predicted 543 customers as churn. This means the model caught more actual churn customers, but precision dropped to 0.519. At threshold 0.70, precision increased to 0.739, but recall dropped to 0.182, and only 92 customers were predicted as churn.

This shows the tradeoff clearly. A lower threshold may be useful if the company has enough budget to contact more customers. A higher threshold may be useful if the company only wants to focus on a smaller group with stronger churn signals.

There is no single best threshold for every business. The choice should depend on customer value, retention cost, and campaign capacity.

## 11. Business Recommendations

Based on the analysis, I would suggest several retention ideas.

First, the company should pay attention to month-to-month contract customers, especially when they also have high predicted churn probability. Renewal incentives or contract upgrade offers may help some of these customers.

Second, early customer experience matters. Customers with short tenure appear more likely to churn, so onboarding support in the first few months could be useful.

Third, customers with high monthly charges should be reviewed carefully. Some may need better bundles, clearer service value, or targeted discounts.

Fourth, payment experience may matter. Electronic check users had higher churn in this dataset. The company could encourage automatic payment methods by making them convenient, not by forcing customers.

Fifth, service support can be used as a retention tool. Customers without tech support or online security may need more support options.

## 12. Limitations

This project has clear limitations. The dataset is public and does not include live company behaviour data. It does not include customer complaints, competitor prices, customer lifetime value, retention campaign history, or actual retention cost.

The models are also basic and only lightly tuned. The purpose is to compare common methods and explain the workflow, not to build the best possible model.

The risk thresholds are simple portfolio-level choices. In a real company, the thresholds should be tested with business data and adjusted by budget and customer value.

The feature importance results show associations in the dataset, but they should not be treated as direct causal proof.

## 13. Conclusion

This project shows a practical customer churn analysis workflow. Gradient Boosting and Logistic Regression performed relatively well in the model comparison. The analysis also showed useful churn-related patterns, including contract type, tenure, monthly charges, payment method, and service support.

The most useful part of the project is not only the model comparison. It is the step from model output to customer risk segmentation and threshold analysis. These steps make the results easier to connect with business decisions.

## 14. Personal Reflection

This project helped me connect data analysis with business thinking. Before doing it, I mainly thought about whether a model had good accuracy. After the project, I better understood why recall, precision, and business cost matter.

I also learned that clear explanation is important. A model can produce numbers, but a portfolio project should explain what those numbers mean and what a company could do with them.

If I continue improving this project, I would like to add more model tuning, customer lifetime value analysis, and a better dashboard. I would also like to test whether the same patterns hold on newer or more detailed customer data.
