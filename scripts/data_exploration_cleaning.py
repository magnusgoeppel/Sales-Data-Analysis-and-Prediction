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

    """
    def find_non_ascii(s):
        if isinstance(s, str):  # Check if input is a string
            non_ascii_chars = [c for c in s if ord(c) >= 128]
            return non_ascii_chars if non_ascii_chars else None
        return None  # Return None if not a string

    # Assuming df is your DataFrame
    # Check for non-ASCII characters in each text column
    for column in df.columns:
        if df[column].dtype == object:  # Apply only to text columns
            df[f'{column}_non_ascii_chars'] = df[column].apply(find_non_ascii)

    # Now check for any row that has non-ASCII characters in any column
    df['contains_non_ascii'] = df.filter(like='_non_ascii_chars').applymap(lambda x: x is not None).any(axis=1)

    # Filter to get only rows that contain non-ASCII characters
    non_ascii_rows = df[df['contains_non_ascii'] == True]

    # Print the filtered rows that contain non-ASCII characters and the non-ASCII characters themselves
    if not non_ascii_rows.empty:
        print("Rows with non-ASCII characters found:")
        for index, row in non_ascii_rows.iterrows():
            print(f"Row {index}:")
            for column in df.columns:
                if df[column].dtype == object:
                    non_ascii_column = f'{column}_non_ascii_chars'
                    if non_ascii_column in df.columns:  # Ensure the column exists
                        non_ascii_chars = row[non_ascii_column]
                        if non_ascii_chars:
                            print(f"Column '{column}': {row[column]}")
                            print(f"Non-ASCII characters: {', '.join(non_ascii_chars)}")
    else:
        print("No rows with non-ASCII characters found.")
    """

    # Adresslinie Split
    return df


def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)
