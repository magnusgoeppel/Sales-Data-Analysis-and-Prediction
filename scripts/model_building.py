from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def hyperparameter_tuning(model, x_train, y_train):
    param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_


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
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Store the metrics in a string
    metrics = (f"Performance Evaluation:\n"
               f"Mean Absolute Error (MAE): {round(mae, 3)}\n"
               f"Mean Squared Error (MSE): {round(mse, 3)}\n"
               f"Root Mean Squared Error (RMSE): {round(rmse, 3)}\n"
               f"R^2 Score: {round(r2, 3)}")

    return metrics
