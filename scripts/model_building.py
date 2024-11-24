import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree


# Hyperparameter tuning
def hyperparameter_tuning(model, x_train, y_train):
    # Define the hyperparameters to tune
    param_grid = {
        'max_depth': [None, 10, 20, 30],  # None: no limit, 10: moderate, 20: high, 30: very high
        'min_samples_split': [2, 5, 10]   # 2: few, 5: moderate, 10: many
    }

    # Initialize the grid search
    grid_search = GridSearchCV(model, param_grid, cv=5)

    # Fit the model with the grid search
    grid_search.fit(x_train, y_train)

    # Return the best estimator
    return grid_search.best_estimator_


# Format the model parameters
def format_model_params(model):
    # Convert to string
    params = str(model.get_params())
    # Format the string
    params = params.replace(',', ',\n')
    params = params.replace('{', '')
    params = params.replace('}', '')
    params = params.replace("'", "")

    # Remove the space after line break
    lines = params.splitlines()
    stripped_lines = [line.strip() for line in lines]
    formatted_params = '\n'.join(stripped_lines)

    return formatted_params


# Build and train the model with k-fold cross validation
def train_model(model, x_train, y_train, k):
    # k-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Initialize lists to store the results of each fold
    y_tests = []
    y_preds = []

    # Do the k-fold cross validation
    for train_index, test_index in kf.split(x_train):
        # Split the data into training and test sets
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model
        model.fit(x_train_fold, y_train_fold)

        # Predict the target variable
        y_pred = model.predict(x_test_fold)

        # Store predictions and test values
        y_tests.extend(y_test_fold)
        y_preds.extend(y_pred)

    return model, y_tests, y_preds


# Performance evaluation
def performance_evaluation(y_test, y_pred):
    # Calculate the metrics
    r2 = r2_score(y_test, y_pred)
    rss = sum((np.array(y_test) - np.array(y_pred)) ** 2)
    mse = mean_squared_error(y_test, y_pred)

    # Store the metrics in a string
    metrics = (f"R^2 Score: {round(r2, 3)}\n"
               f"RSS: {round(rss, 3)}\n"
               f"Mean Squared Error (MSE): {round(mse, 3)}")

    return metrics


# Visualize the Decision Tree
def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(33, 8))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True,
              impurity=True, proportion=True, precision=3, fontsize=13, max_depth=3)
    plt.title("Decision Tree Model", fontsize=35)
    plt.tight_layout()
    plt.show()


# Visualize the actual vs predicted values
def plot_actual_vs_predicted_values(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
    plt.xlabel("Actual Values", fontsize=10)
    plt.ylabel("Predicted Values", fontsize=10)
    plt.title("Actual vs Predicted Values", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
