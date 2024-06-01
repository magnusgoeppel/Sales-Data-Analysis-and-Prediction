import pandas as pd
from scripts.feature_engineering import create_features

# Load the data in a DataFrame
try:
    data = pd.read_csv("data/sales_data_sample.csv", encoding="windows-1252")  # encoding non ascii characters
    df = pd.DataFrame(data)
    print(df.head())
except FileNotFoundError:
    print("File not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# 2. Data exploration and cleaning
# 2.1. Check the data (missing values, data types, etc.)
# 2.2. Clean the data
# 2.3. Explore the data (Histograms, Boxplots, Scatterplots, etc.)
# remove outliers (write down steps)

# 3. Feature engineering (1. step)
df = create_features(df)
# 3.2. Transform the data (e.g. scaling, encoding, etc.)
# --> create a new csv file with all needed features

# 4. Data preprocessing
# 4.1. Scale the data
# --> min max scaling
# 4.2. Split the data into training and test sets
# --> first do the classical approach (80 percent train data and 20 percent test)
# --> and maybe k-fold

# --> Sunday --> meeting


# 5. Model Building
# 5.1. Choose a model
# --> regression trees (compare it with linear regression --> maybe)
# 5.2. Configure the model (e.g. hyperparameters)
# 5.3. Train the model

# 6. Performance evaluation and Visualization
# 6.1. Evaluate the model (e.g. RMSE, R^2, etc.)
# 6.2. Visualize the model (e.g. feature importance, predictions, etc.)


# --> Thursday
# 7. Dokumentation and Presentation
# 7.1. Comment the code
# 7.2. Create the Ananconda environment file (environment.yml)
# 7.3. Write the README file
# 7.4. Create the presentation
