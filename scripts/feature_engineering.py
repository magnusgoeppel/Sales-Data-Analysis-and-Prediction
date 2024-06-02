import pandas as pd


# Create the features
def create_features(df):
    # Select the features
    x = df[["PRICEEACH", "ORDERDATE", "PRODUCTLINE", "MSRP", "COUNTRY", "DEALSIZE"]].copy()

    # Extract the year, month and weekday from the orderdate
    x["YEAR"] = pd.to_datetime(x["ORDERDATE"]).dt.year
    x["MONTH"] = pd.to_datetime(x["ORDERDATE"]).dt.month
    x["WEEKDAY"] = pd.to_datetime(x["ORDERDATE"]).dt.weekday

    """
    # Check for valid years in ORDERDATE
    years_in_data = pd.to_datetime(df["ORDERDATE"]).dt.year.unique()
    years_in_data_str = ', '.join(map(str, years_in_data))
    print(f"Years in data: {years_in_data_str}")
    """

    # Define the holidays
    def is_holiday(date):
        fixed_holidays = [
            '01/01',  # New Year's Day
            '01/06',  # Epiphany
            '05/01',  # National Holiday
            '08/15',  # Assumption Day
            '10/26',  # National Day
            '11/01',  # All Saints' Day
            '12/08',  # Immaculate Conception
            '12/25',  # Christmas Day
            '12/26'  # St. Stephen's Day
        ]

        variable_holidays = {
            2003: [
                '04/21/2003',  # Easter Monday
                '05/29/2003',  # Ascension Day
                '06/09/2003',  # Whit Monday
                '06/19/2003'  # Corpus Christi
            ],
            2004: [
                '04/12/2004',  # Easter Monday
                '05/20/2004',  # Ascension Day
                '05/31/2004',  # Whit Monday
                '06/10/2004'  # Corpus Christi
            ],
            2005: [
                '03/28/2005',  # Easter Monday
                '05/05/2005',  # Ascension Day
                '05/16/2005',  # Whit Monday
                '05/26/2005'  # Corpus Christi
            ]
        }

        # Extract the year and date string
        year = date.year
        date_str = date.strftime('%m/%d')
        full_date_str = date.strftime('%m/%d/%Y')

        # Check fixed holidays
        if date_str in fixed_holidays:
            return True

        # Check variable holidays
        if year in variable_holidays and full_date_str in variable_holidays[year]:
            return True

        return False

    # Apply the is_holiday function to create the IS_HOLIDAY column
    x["IS_HOLIDAY"] = pd.to_datetime(df["ORDERDATE"]).apply(is_holiday)

    # Drop the orderdate column
    x.drop(columns=["ORDERDATE"], inplace=True)

    return x


def save_transformed_data(x, y, file_path):
    # Combine features and target column
    transformed_df = pd.concat([x, y], axis=1)
    transformed_df.to_csv(file_path, index=False)