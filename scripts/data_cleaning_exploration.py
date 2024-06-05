import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_and_clean_data(df):
    # Missing values
    missing_values = df.isnull().sum()

    if missing_values.any():
        df = df.replace("", np.nan)

    # Splite ORDERDATE into date and time
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"]).dt.date
    df["ORDERTIME"] = pd.to_datetime(df["ORDERDATE"]).dt.time

    # Check if ORDERTIME column contains only times set to '00:00:00'
    if (df['ORDERTIME'] == pd.to_datetime('00:00:00').time()).all():
        df = df.drop(columns=['ORDERTIME'])

    # Check for duplicates
    duplicates = df.duplicated().sum()

    if duplicates:
        df = df.drop_duplicates()

    # Check for non ascii characters
    non_ascii_chars_dict = {}

    # Iterating over all columns and rows to find non-ascii characters
    for col in df.columns:
        for row in df.index:
            value = str(df.at[row, col])
            for char in value:
                if ord(char) > 127:
                    if char not in non_ascii_chars_dict:
                        non_ascii_chars_dict[char] = (value, col, row)

    """
    # Print non-ascii characters
    print("Non-Ascii characters:")
    for char, (value, col, row) in non_ascii_chars_dict.items():
        print(f"Row: {row+1}, {col}: {value}, Character: {char}")
    """

    # Replace Non-Ascii characters
    replacement_list = ['ae', 'y', 'i']
    replacement_dict = {}

    # Assign replacements from the list to the non-ASCII characters
    for i, char in enumerate(non_ascii_chars_dict.keys()):
        if i < len(replacement_list):
            replacement_dict[char] = replacement_list[i]

    # Replace all non-ascii characters in the DataFrame
    for col in df.columns:
        for row in df.index:
            value = str(df.at[row, col])
            new_value_chars = []
            for char in value:
                new_char = replacement_dict.get(char, char)
                new_value_chars.append(new_char)
            new_value = ''.join(new_value_chars)
            df.at[row, col] = new_value

    # Assign data types to the columns
    numeric_cols = ['QUANTITYORDERED', 'PRICEEACH', 'SALES', 'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'MSRP']
    categorical_cols = ['ORDERNUMBER', 'ORDERLINENUMBER', 'ORDERDATE', 'STATUS', 'PRODUCTLINE', 'PRODUCTCODE',
                        'CUSTOMERNAME', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE',
                        'COUNTRY', 'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME', 'DEALSIZE']

    # Iterate over columns and assign data types
    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        elif col in categorical_cols:
            df[col] = df[col].astype('category')

    return df


# Explore the data
def explore_data(df):
    # Numeric Summaries
    numeric_summary = df.describe()
    print("\033[1mNumeric Summaries:\033[0m")
    print(round(numeric_summary, 3))
    print()

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Boxplots
    numeric_cols = df.select_dtypes(include=['float64']).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Boxplots of Numeric Columns', fontsize=20)
    plt.show()

    # Scatterplots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        if col != 'QUANTITYORDERED':
            plt.subplot(3, 3, i + 1)
            sns.scatterplot(x=df[col], y=df['QUANTITYORDERED'])
            plt.title(f'Scatterplot of {col} vs QUANTITYORDERED')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Scatterplots of Numeric Columns vs QUANTITYORDERED', fontsize=20)
    plt.show()

    # Density Plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(3, 3, i + 1)
        sns.kdeplot(df[col], fill=True)
        plt.title(f'Density Plot of {col}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Density Plots of Numeric Columns', fontsize=20)
    plt.show()
