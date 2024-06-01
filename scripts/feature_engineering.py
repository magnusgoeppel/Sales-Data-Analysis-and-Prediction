import pandas as pd


# Create the features
def create_features(df):
    # Convert ORDERDATE to datetime
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # ...
