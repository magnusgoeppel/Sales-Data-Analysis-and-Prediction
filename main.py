import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from scripts.data_cleaning_exploration import check_and_clean_data, explore_data
from scripts.data_preprocessing import create_features, save_transformed_data, encode, scale_x, scale_y
from scripts.model_building import train_model, performance_evaluation, hyperparameter_tuning, plot_decision_tree, \
    plot_actual_vs_predicted_values, format_model_params

# Display all columns
pd.set_option('display.max_columns', None)


# 1. Data acquisition (load the data)
data = pd.read_csv("data/sales_data_sample.csv", encoding="latin1")  # encoding non ascii characters
df = pd.DataFrame(data)

# 2. Data exploration and cleaning

# 2.1. Check and clean the data
df = check_and_clean_data(df)

# 2.2. Save the cleaned data to a new CSV file
df.to_csv("data/cleaned_sales_data.csv", index=False)

# 2.3. Explore the data
explore_data(df)


# 3. Data preprocessing

# 3.1. Feature and target selection
x = create_features(df)
y = df["QUANTITYORDERED"]

# 3.2. Save the transformed data to a new CSV file
save_transformed_data(x, y, "data/transformed_sales_data.csv")

# 3.3. Encode non-numeric data
x = encode(x)

# 3.4. Scale the data
x = scale_x(x)
y = scale_y(y)

# 3.5. Split the data into training and test sets (80% training, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3.6. Split the training data into training and validation sets (65% training, 15% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1875, random_state=42)


# 4. Model Building

# 4.1. Select Regression Tree as the model
model = DecisionTreeRegressor(random_state=42)

# 4.2. Hyperparameter tuning
model = hyperparameter_tuning(model, x_val, y_val)

# 4.3. Model parameters
formatted_params = format_model_params(model)
print(f"\033[1mModel Parameters:\033[0m\n{formatted_params}\n")

# 4.4. Build and train the model with 5-fold cross validation
model, y_tests, y_preds = train_model(model, x_train, y_train, 5)

# 4.5. Performance evaluation
metrics = performance_evaluation(y_tests, y_preds)
print(f"\033[1mPerformance evaluation:\033[0m\n{metrics}")

# 4.5. Visualize the model
plot_decision_tree(model, x.columns)
plot_actual_vs_predicted_values(y_tests, y_preds)


# 5. Dokumentation and Presentation
# 5.1. Create the Ananconda environment file (environment.yml)
# 5.2. Write the README file
# 5.3. Create the presentation
