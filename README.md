# Sales Data Analysis and Prediction
This project focuses on analyzing and predicting sales data using a Decision Tree Regression model. 
The project includes steps for data acquisition, cleaning, exploration, preprocessing, and model building, 
along with performance evaluation and visualization.

## Dataset
Link to Kaggle: [Sample Sales Data](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

## Project Structure
The project is structured into several scripts and steps, each performing specific tasks in the data analysis 
and model building process.

### Data Acquisition
The sales data is loaded from a CSV file and converted into a Pandas DataFrame.

### Data Cleaning and Exploration
- Data Cleaning: The data is checked for missing values, duplicates, and non-ASCII characters. Appropriate data types 
are assigned to each column.
- Data Exploration: Descriptive statistics and visualizations (box plots, scatter plots, density plots) are generated 
- to understand the data distribution and relationships.

### Data Preprocessing
- Feature and Target Selection: Features are created from the data, including extracting year, month, and weekday from 
the order date and identifying holidays.
- Encoding: Categorical data is encoded using one-hot encoding and ordinal encoding.
- Scaling: Feature and target variables are scaled using MinMaxScaler.
- Data Splitting: The data is split into training, validation, and test sets.

### Model Building
- Model Selection: A Decision Tree Regressor is chosen as the model.
- Hyperparameter Tuning: Grid search is used to find the best hyperparameters for the model.
- Model Training: The model is trained using 5-fold cross-validation.
- Performance Evaluation: The model's performance is evaluated using R² score, RSS, and MSE metrics.
- Visualization: The decision tree and actual vs predicted values are visualized.

## Anaconda Environment
To ensure that all necessary dependencies are installed and the environment is correctly set up, 
create an environment.yml file for an Anaconda environment. 

## Requirements
- Python 3.8
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn or Anaconda environment

## Running the Project
- Start the main script to execute the data analysis and model building process:
```python main.py```

## Developers

4th Semester Project at University of Applied Sciences Technikum Wien 
for the course "Data Science and Machine Learning" from:

- Filipson Evelina 
- Garbuzov Vladimir 
- Göppel Magnus 
- Petkovic Vladan