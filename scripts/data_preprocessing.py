import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Scale x
def scale_x(df):
    # Select only float64 columns
    float_cols = df.select_dtypes(include=['float64']).columns

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the selected columns
    df[float_cols] = scaler.fit_transform(df[float_cols])

    # Return the DataFrame with scaled float64 columns
    return df


# scale_y
def scale_y(y):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    # Return as Series
    return pd.Series(y_scaled.flatten())
