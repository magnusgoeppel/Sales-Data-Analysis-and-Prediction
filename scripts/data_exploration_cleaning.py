import numpy as np
import pandas as pd


def check_and_clean_data(df):
    # Missing values
    missing_values = df.isnull().sum()

    if missing_values.any():
        df = df.replace("", np.nan)

    # Splite ORDERDATE into date and time
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"]).dt.date
    df["ORDERTIME"] = pd.to_datetime(df["ORDERDATE"]).dt.time

    # Check ORDERTIME for all zeros
    if (df['ORDERTIME'] == pd.to_datetime('00:00:00').time()).all():
        df = df.drop(columns=['ORDERTIME'])

    # Check for duplicates
    duplicates = df.duplicated().sum()

    if duplicates:
        df = df.drop_duplicates()

    # Check for non ascii characters

    # Dictionary to store non-ascii characters
    non_ascii_chars_dict = {}

    # Iterating over all columns and rows to find non-ascii characters
    for col in df.columns:
        for row in df.index:
            value = str(df.at[row, col])
            for char in value:
                if ord(char) > 127:
                    if char not in non_ascii_chars_dict:
                        non_ascii_chars_dict[char] = (value, col, row)

    # Print non-ascii characters
    for char, (value, col, row) in non_ascii_chars_dict.items():
        print(f"Row: {row+1}, {col}: {value}, Character: {char}")

    return df


def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)
