import os
import pandas as pd

output_folder = "data/merged"

# Read the CSV files into DataFrames
df_cases = pd.read_csv('data/cleaned/cleaned_confirmed_cases.csv')
df_deaths = pd.read_csv('data/cleaned/cleaned_deaths_long.csv')
df_economic = pd.read_csv('data/cleaned/cleaned_economic_indicators_long.csv')
df_recovered = pd.read_csv('data/cleaned/cleaned_recovered_long.csv')
df_vaccine = pd.read_csv('data/cleaned/cleaned_vaccine_long.csv')
df_vaccine_doses = pd.read_csv('data/cleaned/cleaned_vaccine_doses_long.csv')

# --- Data Preparation ---
# Convert `Date` columns to datetime objects
df_cases['Date'] = pd.to_datetime(df_cases['Date'])
df_deaths['Date'] = pd.to_datetime(df_deaths['Date'])
df_recovered['Date'] = pd.to_datetime(df_recovered['Date'])
df_vaccine['Date'] = pd.to_datetime(df_vaccine['Date'])
df_vaccine_doses['Date'] = pd.to_datetime(df_vaccine_doses['Date'])

# Pivot the economic DataFrame to have indicators as columns
df_economic_pivot = df_economic.pivot_table(index=['Country/Region', 'Year'], columns='Series Name', values='Value').reset_index()

# --- Merging DataFrames ---
# Merge the COVID related DataFrames (cases, deaths, recovered, vaccine)
# Start with merging cases and deaths
df_merged = pd.merge(df_cases, df_deaths, on=['Country/Region', 'Date'], how='outer')
# Merge with recovered data
#df_merged = pd.merge(df_merged, df_recovered, on=['Country/Region', 'Date'], how='outer')
# Merge with vaccine data
df_merged = pd.merge(df_merged, df_vaccine, on=['Country/Region', 'Date'], how='outer')
# Merge with population data
df_merged = pd.merge(df_merged, df_vaccine_doses, on=['Country/Region', 'Date'], how='outer')
      
# Extract Year from the Date column in the merged COVID DataFrame
df_merged['Year'] = df_merged['Date'].dt.year

# Merge the combined COVID data with the pivoted economic data
df_combined = pd.merge(df_merged, df_economic_pivot, on=['Country/Region', 'Year'], how='outer')
#df_combined = df_combined.fillna('NaN')
# Display columns and their data types
print("\nCombined DataFrame Information:")
print(df_combined.info())

# Display the shape of the combined DataFrame
print("\nShape of the Combined DataFrame:")
print(f"Rows: {df_combined.shape[0]}, Columns: {df_combined.shape[1]}")

# Display the date range in the combined DataFrame
min_date = df_combined['Date'].dropna().min()
max_date = df_combined['Date'].dropna().max()
df_combined = df_combined[~df_combined['Country/Region'].isin(['Winter Olympics 2022', 'Summer Olympics 2020', 'Diamond Princess', 'World', 'American Samoa', 'Aruba', 'Bermuda', 'Cayman Islands', 'Channel Islands', 'Curacao', 'Faroe Islands', 'French Polynesia', 'Greenland', 'Guam', 'Hong Kong SAR, China', 'Isle of Man', 'Kyrgyz Republic', 'Macao SAR, China', 'Myanmar', 'New Caledonia', 'Northern Mariana Islands', 'Puerto Rico', 'Sint Maarten (Dutch part)', 'St. Martin (French part)', 'Turkmenistan', 'Turks and Caicos Islands', 'Virgin Islands (U.S.)', 'British Virgin Islands', 'Gibraltar', "Korea, Dem. People's Rep.", 'Yemen, Rep.' ])]
df_combined = df_combined.drop(columns=['Year'])
#df_combined = df_combined.dropna()
percentage_columns = [
    'GDP growth (annual %)', 
    'Inflation, consumer prices (annual %)',
    'Unemployment, total (% of total labor force) (national estimate)'
]
for col in percentage_columns:
    # First ensure the column is numeric
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    # Then divide by 100
    df_combined[col] = df_combined[col] / 100
print(f"\nDate range in Combined DataFrame: {min_date} to {max_date}")
df_combined.to_csv(os.path.join(output_folder, "combined_dataset.csv"), index=False)

