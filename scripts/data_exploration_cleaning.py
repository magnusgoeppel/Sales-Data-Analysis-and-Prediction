from main import df

# Find non-ASCII characters in the DataFrame
def contains_non_ascii(s):
    if isinstance(s, str):  # Check if input is a string
        return any(ord(c) >= 128 for c in s)
    return False  # Return False if not a string

# Create a new column for each text-based column to check for non-ASCII characters
for column in df.columns:
    if df[column].dtype == object:  # Filters to apply only to text columns
        df[f'{column}_contains_non_ascii'] = df[column].apply(contains_non_ascii)

# Now check for any row that has non-ASCII characters in any column
df['contains_non_ascii'] = df.filter(like='_contains_non_ascii').any(axis=1)

# Filter to get only rows that contain non-ASCII characters
non_ascii_rows = df[df['contains_non_ascii'] == True]
print(non_ascii_rows)