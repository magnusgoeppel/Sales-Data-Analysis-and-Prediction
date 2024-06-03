import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.data_exploration_cleaning import check_and_clean_data
from scripts.data_exploration_cleaning import explore_data
from scripts.data_preprocessing import create_features
from scripts.data_preprocessing import save_transformed_data
from scripts.data_preprocessing import encode
from scripts.data_preprocessing import scale_x
from scripts.data_preprocessing import scale_y

# To display all columns
pd.set_option('display.max_columns', None)


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


# 5. Model Building
# 5.1. Choose a model
# --> regression trees (compare it with linear regression --> maybe)
# 5.2. Configure the model (e.g. hyperparameters)
# 5.3. Train the model
# 5.4. (k-fold)
# 5.5. Evaluate the model (e.g. RMSE, R^2, etc.)
# 5.6. Visualize the model (e.g. feature importance, predictions, etc.)


# --> Thursday
# 6. Dokumentation and Presentation
# 6.1. Comment the code
# 6.2. Create the Ananconda environment file (environment.yml)
# 6.3. Write the README file
# 6.4. Create the presentation
