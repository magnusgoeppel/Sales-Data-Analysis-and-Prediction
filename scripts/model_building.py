import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Build and train the model
def build_and_train_model(model, x_train, y_train, k):
    # k-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize lists to store the results of each fold
    y_preds = []
    y_tests = []

    # Fit the model and predict the target variable
    for train_index, test_index in kf.split(x_train):
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_test_fold)

        # Ensure indices are reset to avoid indexing issues
        y_test_fold = y_test_fold.reset_index(drop=True)

        # Store predictions and actual values
        y_preds.extend(y_pred)

    # Convert to numpy arrays for evaluation
    y_preds = np.array(y_preds)
    y_tests = np.array(y_tests)

    return model, y_preds


# Performance evaluation
def performance_evaluation(y_test, y_pred):
    # Calculate the metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Store the metrics in a dictionary
    metrics = {"Mean Absolute Error": mae,
               "Mean Squared Error": mse,
               "Root Mean Squared Error": rmse,
               "R^2 Score": r2}

    return metrics
