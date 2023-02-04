# Loan-Eligibility
This script utilizes a logistical regression to predict if we should give a loan to a person

Documentation
This code imports the necessary libraries including numpy, pandas, missingno, and matplotlib.

The loan-train and loan-test datasets are read using pandas and stored in the variables loan_train and loan_test respectively.

The missing values in the datasets are then filled using either the mode (most frequent value) or the mean for the specific columns.

Categorical variables in the datasets are transformed into numerical values for processing by the machine learning model.

A Logistic Regression model is imported from the sklearn library and trained on the training data.

The accuracy of the model is then computed and the coefficients and intercepts are printed.
