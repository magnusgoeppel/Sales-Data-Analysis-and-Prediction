import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Scale x
def scale_x(x):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    x_scaled = scaler.fit_transform(x)

    # Return as DataFrame
    return pd.DataFrame(x_scaled, columns=x.columns)


# scale_y
def scale_y(y):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    # Return as Series
    return pd.Series(y_scaled.flatten())
