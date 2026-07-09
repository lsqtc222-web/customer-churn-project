# Customer Churn Analysis: Business Notes

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
