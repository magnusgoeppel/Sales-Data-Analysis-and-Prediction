import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


# Create the features
def create_features(df):
    # Select the features
    x = df[["PRICEEACH", "ORDERDATE", "PRODUCTLINE", "MSRP", "COUNTRY", "DEALSIZE"]].copy()

    # Extract the year, month and weekday from the orderdate
    x["YEAR"] = pd.to_datetime(x["ORDERDATE"]).dt.year.astype('float64')
    x["MONTH"] = pd.to_datetime(x["ORDERDATE"]).dt.month.astype('float64')
    x["WEEKDAY"] = pd.to_datetime(x["ORDERDATE"]).dt.weekday.astype('float64')

    """
    # Check for valid years in ORDERDATE
    years_in_data = pd.to_datetime(df["ORDERDATE"]).dt.year.unique()
    years_in_data_str = ', '.join(map(str, years_in_data))
    print(f"Years in data: {years_in_data_str}")
    """

    """
    # Check Country in the data
    countries = x["COUNTRY"].unique()
    countries_str = ', '.join(map(str, countries))
    print(f"Countries: {countries_str}")
    """

    # Define the holidays
    def is_holiday(date, country):
        holidays = {
            'USA': {
                'fixed': ['01/01', '07/04', '11/11', '12/25'],
                'variable': {
                    2003: ['01/20/2003', '02/17/2003', '05/26/2003', '09/01/2003', '10/13/2003', '11/27/2003'],
                    2004: ['01/19/2004', '02/16/2004', '05/31/2004', '09/06/2004', '10/11/2004', '11/25/2004'],
                    2005: ['01/17/2005', '02/21/2005', '05/30/2005', '09/05/2005', '10/10/2005', '11/24/2005']
                }
            },
            'France': {
                'fixed': ['01/01', '05/01', '05/08', '07/14', '08/15', '11/01', '11/11', '12/25'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003', '06/09/2003'],
                    2004: ['04/12/2004', '05/20/2004', '05/31/2004'],
                    2005: ['03/28/2005', '05/05/2005', '05/16/2005']
                }
            },
            'Norway': {
                'fixed': ['01/01', '05/01', '05/17', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003', '06/09/2003'],
                    2004: ['04/12/2004', '05/20/2004', '05/31/2004'],
                    2005: ['03/28/2005', '05/05/2005', '05/16/2005']
                }
            },
            'Australia': {
                'fixed': ['01/01', '01/26', '04/25', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '06/09/2003'],
                    2004: ['04/12/2004', '06/14/2004'],
                    2005: ['03/28/2005', '06/13/2005']
                }
            },
            'Finland': {
                'fixed': ['01/01', '05/01', '12/06', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003'],
                    2004: ['04/12/2004', '05/20/2004'],
                    2005: ['03/28/2005', '05/05/2005']
                }
            },
            'Austria': {
                'fixed': ['01/01', '01/06', '05/01', '08/15', '10/26', '11/01', '12/08', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003', '06/09/2003', '06/19/2003'],
                    2004: ['04/12/2004', '05/20/2004', '05/31/2004', '06/10/2004'],
                    2005: ['03/28/2005', '05/05/2005', '05/16/2005', '05/26/2005']
                }
            },
            'UK': {
                'fixed': ['01/01', '12/25', '12/26'],
                'variable': {
                    2003: ['04/18/2003', '04/21/2003', '05/05/2003', '05/26/2003', '08/25/2003'],
                    2004: ['04/09/2004', '04/12/2004', '05/03/2004', '05/31/2004', '08/30/2004'],
                    2005: ['03/25/2005', '03/28/2005', '05/02/2005', '05/30/2005', '08/29/2005']
                }
            },
            'Spain': {
                'fixed': ['01/01', '01/06', '05/01', '08/15', '10/12', '11/01', '12/06', '12/08', '12/25'],
                'variable': {
                    2003: ['04/18/2003'],
                    2004: ['04/09/2004'],
                    2005: ['03/25/2005']
                }
            },
            'Sweden': {
                'fixed': ['01/01', '01/06', '05/01', '06/06', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003'],
                    2004: ['04/12/2004', '05/20/2004'],
                    2005: ['03/28/2005', '05/05/2005']
                }
            },
            'Singapore': {
                'fixed': ['01/01', '05/01', '08/09', '12/25'],
                'variable': {
                    2003: ['02/01/2003', '02/02/2003', '04/18/2003', '12/25/2003'],
                    2004: ['01/22/2004', '01/23/2004', '04/09/2004', '11/14/2004'],
                    2005: ['02/09/2005', '02/10/2005', '03/25/2005', '11/03/2005']
                }
            },
            'Canada': {
                'fixed': ['01/01', '07/01', '12/25', '12/26'],
                'variable': {
                    2003: ['04/18/2003', '04/21/2003', '05/19/2003', '10/13/2003'],
                    2004: ['04/09/2004', '04/12/2004', '05/24/2004', '10/11/2004'],
                    2005: ['03/25/2005', '03/28/2005', '05/23/2005', '10/10/2005']
                }
            },
            'Japan': {
                'fixed': ['01/01', '02/11', '04/29', '05/03', '05/04', '05/05', '11/03', '11/23', '12/23'],
                'variable': {
                    2003: ['01/13/2003', '07/21/2003', '09/15/2003', '10/13/2003'],
                    2004: ['01/12/2004', '07/19/2004', '09/20/2004', '10/11/2004'],
                    2005: ['01/10/2005', '07/18/2005', '09/19/2005', '10/10/2005']
                }
            },
            'Italy': {
                'fixed': ['01/01', '01/06', '04/25', '05/01', '06/02', '08/15', '11/01', '12/08', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003'],
                    2004: ['04/12/2004'],
                    2005: ['03/28/2005']
                }
            },
            'Denmark': {
                'fixed': ['01/01', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003'],
                    2004: ['04/12/2004', '05/20/2004'],
                    2005: ['03/28/2005', '05/05/2005']
                }
            },
            'Belgium': {
                'fixed': ['01/01', '05/01', '07/21', '08/15', '11/01', '11/11', '12/25'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003', '06/09/2003'],
                    2004: ['04/12/2004', '05/20/2004', '05/31/2004'],
                    2005: ['03/28/2005', '05/05/2005', '05/16/2005']
                }
            },
            'Philippines': {
                'fixed': ['01/01', '06/12', '12/25', '12/30'],
                'variable': {
                    2003: ['04/17/2003', '04/18/2003', '11/25/2003'],
                    2004: ['04/08/2004', '04/09/2004', '11/14/2004'],
                    2005: ['03/24/2005', '03/25/2005', '11/03/2005']
                }
            },
            'Germany': {
                'fixed': ['01/01', '05/01', '10/03', '12/25', '12/26'],
                'variable': {
                    2003: ['04/18/2003', '04/21/2003', '05/29/2003'],
                    2004: ['04/09/2004', '04/12/2004', '05/20/2004'],
                    2005: ['03/25/2005', '03/28/2005', '05/05/2005']
                }
            },
            'Switzerland': {
                'fixed': ['01/01', '08/01', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/29/2003'],
                    2004: ['04/12/2004', '05/20/2004'],
                    2005: ['03/28/2005', '05/05/2005']
                }
            },
            'Ireland': {
                'fixed': ['01/01', '03/17', '12/25', '12/26'],
                'variable': {
                    2003: ['04/21/2003', '05/05/2003', '06/02/2003', '08/04/2003'],
                    2004: ['04/12/2004', '05/03/2004', '06/07/2004', '08/02/2004'],
                    2005: ['03/28/2005', '05/02/2005', '06/06/2005', '08/01/2005']
                }
            }
        }
        fixed_holidays = holidays[country]['fixed']
        variable_holidays = holidays[country]['variable']

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
    x["IS_HOLIDAY"] = False

    for index, row in x.iterrows():
        x.at[index, "IS_HOLIDAY"] = is_holiday(pd.to_datetime(row["ORDERDATE"]), row["COUNTRY"])

    # x['IS_HOLIDAY'] = x['IS_HOLIDAY'].astype('category')

    # Drop the orderdate column
    x.drop(columns=["ORDERDATE"], inplace=True)

    return x


def save_transformed_data(x, y, file_path):
    # Combine features and target column
    transformed_df = pd.concat([x, y], axis=1)
    transformed_df.to_csv(file_path, index=False)


def encode(df):
    # One hot encoding for PRODUCTLINE
    hot_one_encoder = OneHotEncoder()
    encoded_productline = hot_one_encoder.fit_transform(df[["PRODUCTLINE"]]).toarray()
    encoded_productline_df = pd.DataFrame(encoded_productline, columns=hot_one_encoder.get_feature_names_out(["PRODUCTLINE"]))

    # One hot encoding for COUNTRY
    encoded_country = hot_one_encoder.fit_transform(df[["COUNTRY"]]).toarray()
    encoded_country_df = pd.DataFrame(encoded_country, columns=hot_one_encoder.get_feature_names_out(["COUNTRY"]))

    """
    # Check the categories for DEALSIZE
    deal_size_categories = df["DEALSIZE"].unique()
    deal_size_categories_str = ', '.join(map(str, deal_size_categories))
    print(f"Deal Size categories: {deal_size_categories_str}")
    """

    # Ordinal encoding for DEALSIZE
    ordinal_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
    encoded_dealsize = ordinal_encoder.fit_transform(df[["DEALSIZE"]])
    encoded_dealsize_df = pd.DataFrame(encoded_dealsize, columns=["DEALSIZE"])

    # Drop the original columns
    df = df.drop(columns=["PRODUCTLINE", "COUNTRY", "DEALSIZE"], axis=1)

    # Combine the encoded columns
    df_encoded = pd.concat([df, encoded_productline_df, encoded_country_df, encoded_dealsize_df], axis=1)

    # Encode IS_HOLIDAY as float64
    df_encoded['IS_HOLIDAY'] = df_encoded['IS_HOLIDAY'].astype('float64')

    return df_encoded


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


# Scale y
def scale_y(y):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    # Return as Series
    return pd.Series(y_scaled.flatten())
