# Sales Data Analysis and Prediction

This project focuses on analyzing and predicting sales data using a Decision Tree Regression model. 
The project includes steps for data acquisition, cleaning, exploration, preprocessing, and model building, 
along with performance evaluation and visualization.

## Dataset

The dataset used in this project is a sample sales data available on Kaggle. It contains information about orders, 
including order date, quantity ordered, price, product line, and other relevant details.

Link to Kaggle: [Sample Sales Data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

## Project Structure

The project is structured into several scripts and steps, each performing specific tasks in the data analysis 
and model building process.

### Data Acquisition

- The sales data is loaded from a CSV file and converted into a Pandas DataFrame.

### Data Cleaning and Exploration

- **Data Cleaning:** The data is checked for missing values, duplicates, and non-ASCII characters. 
Appropriate data types are assigned to each column.
- **Data Exploration:** Descriptive statistics and visualizations (box plots, scatter plots, density plots) 
are generated to understand the data distribution and relationships.

### Data Preprocessing

- **Feature and Target Selection:** Features are created from the data, including extracting year, month, and 
weekday from the order date and identifying holidays.
  - **Features:**
    - `PRICEEACH`: The price of each item.
    - `ORDERDATE`: The date when the order was placed.
    - `PRODUCTLINE`: The product line of the item.
    - `MSRP`: Manufacturer's suggested retail price.
    - `COUNTRY`: The country where the order was placed.
    - `DEALSIZE`: The size of the deal (Small, Medium, Large).
    - Additional features derived from `ORDERDATE`:
      - `YEAR`: The year the order was placed.
      - `MONTH`: The month the order was placed.
      - `WEEKDAY`: The weekday the order was placed.
      - `IS_HOLIDAY`: Whether the order date was a holiday in the respective country.
  - **Target:**
    - `QUANTITYORDERED`: The quantity of items ordered.
- **Encoding:** Categorical data is encoded using one-hot encoding and ordinal encoding.
- **Scaling:** Feature and target variables are scaled using MinMaxScaler.
- **Data Splitting:** The data is split into training, validation, and test sets.

### Model Building

- **Model Selection:** A Decision Tree Regressor is chosen as the model.
- **Hyperparameter Tuning:** Grid search is used to find the best hyperparameters for the model.
- **Model Training:** The model is trained using 5-fold cross-validation.
- **Performance Evaluation:** The model's performance is evaluated using R² score, RSS, and MSE metrics.
- **Visualization:** The decision tree and actual vs predicted values are visualized.

## Anaconda Environment

To ensure that all necessary dependencies are installed and the environment is correctly set up, 
create an `environment.yml` file for an Anaconda environment.

## Requirements

- Python 3.8
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn 
  or a Anaconda environment

## Running the Project

Start the main script to execute the data analysis and model building process:

```bash
python main.py
```