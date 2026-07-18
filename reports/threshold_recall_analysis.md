# Threshold and Recall Analysis

## Purpose

This script checks how the churn prediction result changes when the probability threshold is adjusted. The model is the Logistic Regression pipeline used in the main modeling workflow. I used Logistic Regression here because it is easier to explain and it had relatively good recall in the model comparison.

## Results

The threshold comparison is saved in `reports/threshold_comparison.csv`.

At the default threshold of 0.50, the model reached precision of 0.657, recall of 0.559, F1-score of 0.604, and accuracy of 0.806. It predicted 318 customers in the test set as churn.

The lowest tested threshold, 0.30, gave the highest recall of 0.754. This means the model caught more actual churn customers, but it also flagged more customers as high risk. In a real retention campaign, this could increase false positives and campaign cost.

## Business Meaning

In churn prediction, recall matters because missing a high-risk customer may mean losing a chance to contact the customer before they leave. However, precision also matters because retention offers, service calls, and discounts all cost money.

A lower threshold may be useful when the company has enough retention budget and wants to reach more possible churn customers. A higher threshold may be better when the company wants to contact fewer customers with stronger churn signals.

There is no single threshold that is always best. The final choice should depend on business cost, customer value, and the size of the retention budget.
