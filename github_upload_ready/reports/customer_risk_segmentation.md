# Customer Risk Segmentation

## Purpose

After building a churn model, I used the predicted churn probability to split customers into three groups. This makes the model easier to use for business decisions.

- Low Risk: predicted churn probability below 0.30
- Medium Risk: predicted churn probability from 0.30 to 0.60
- High Risk: predicted churn probability above 0.60

## Results

The customer-level scored data is saved in `reports/customer_risk_scored.csv`.

The summary by risk segment is saved in `reports/risk_segment_summary.csv`.

In this result, the high-risk group has a much higher actual churn rate and a shorter average tenure. This is a useful check because the groups are not only different in model score; they also show different real churn behaviour.

## Business Meaning

The segmentation can help a company decide where to spend retention effort. Low-risk customers may only need normal service quality. Medium-risk customers can be monitored. High-risk customers should be contacted earlier, especially when they also have high monthly charges or short tenure.

## Suggested Actions

1. Low Risk: keep regular service quality and avoid unnecessary discounts.
2. Medium Risk: watch for changes in usage or payment behaviour.
3. High Risk: use targeted calls, better onboarding, renewal incentives, or service support.

## Limitations

The 0.30 and 0.60 thresholds are simple choices for this portfolio project. In a real company, I would adjust them by customer lifetime value, retention cost, and campaign budget.
