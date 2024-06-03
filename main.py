import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from scripts.data_exploration_cleaning import check_and_clean_data, explore_data
from scripts.data_preprocessing import create_features, save_transformed_data, encode, scale_x, scale_y
from scripts.model_building import train_model, performance_evaluation, hyperparameter_tuning

# To display all columns
# pd.set_option('display.max_columns', None)


# 1. Data acquisition (load the data)
data = pd.read_csv("data/sales_data_sample.csv", encoding="latin1")  # encoding non ascii characters
df = pd.DataFrame(data)


# 2. Data exploration and cleaning

# 2.1. Check and clean the data
df = check_and_clean_data(df)

# 2.2. Save the cleaned data to a new CSV file
df.to_csv("data/cleaned_sales_data.csv", index=False)

# 2.3. Explore the data (Histograms, Boxplots, Scatterplots)
# explore_data(df)

# 2.4. (remove outliers)


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

# 3.5. Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 4. Model Building

# 4.1. Select Regression Tree as the model
model = DecisionTreeRegressor(random_state=42)

# 4.2. Hyperparameter tuning
model = hyperparameter_tuning(model, x_train, y_train)

# 4.3. Build and train the model with 5-fold cross validation
model, y_tests, y_preds = train_model(model, x_train, y_train, 5)

# 4.4. Performance evaluation
metrics = performance_evaluation(y_tests, y_preds)
print(metrics)

# 4.5. Visualize the model (e.g. feature importance, predictions, etc.)


# --> Thursday
# 5. Dokumentation and Presentation
# 5.1. Comment the code
# 5.2. Create the Ananconda environment file (environment.yml)
# 5.3. Write the README file
# 5.4. Create the presentation
