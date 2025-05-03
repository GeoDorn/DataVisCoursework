import os
import pandas as pd

output_folder = "data/merged"

# Read the CSV files into DataFrames
df_cases = pd.read_csv('data/cleaned/cleaned_confirmed_cases.csv')
df_deaths = pd.read_csv('data/cleaned/cleaned_deaths_long.csv')
df_economic = pd.read_csv('data/cleaned/cleaned_economic_indicators_long.csv')
df_recovered = pd.read_csv('data/cleaned/cleaned_recovered_long.csv')
df_vaccine = pd.read_csv('data/cleaned/cleaned_vaccine_long.csv')

# --- Data Preparation ---
# Convert `Date` columns to datetime objects
df_cases['Date'] = pd.to_datetime(df_cases['Date'])
df_deaths['Date'] = pd.to_datetime(df_deaths['Date'])
df_recovered['Date'] = pd.to_datetime(df_recovered['Date'])
df_vaccine['Date'] = pd.to_datetime(df_vaccine['Date'])

# Pivot the economic DataFrame to have indicators as columns
df_economic_pivot = df_economic.pivot_table(index=['Country/Region', 'Year'], columns='Series Name', values='Value').reset_index()
if 'Country/Region' in df_economic_pivot.columns:
      df_economic_pivot.loc[df_economic_pivot['Country/Region'] == 'Venezuela RB', 'Country/Region'] = 'Venezuela'
      df_economic_pivot.loc[df_economic_pivot['Country/Region'] == 'Korea, Rep', 'Country/Region'] = 'South Korea'
      df_economic_pivot.loc[df_economic_pivot['Country/Region'] == "Korea, Dem. People's Rep.", 'Country/Region'] = 'North Korea'
# --- Merging DataFrames ---
# Merge the COVID related DataFrames (cases, deaths, recovered, vaccine)
# Start with merging cases and deaths
df_merged = pd.merge(df_cases, df_deaths, on=['Country/Region', 'Date'], how='outer')
# Merge with recovered data
df_merged = pd.merge(df_merged, df_recovered, on=['Country/Region', 'Date'], how='outer')
# Merge with vaccine data
df_merged = pd.merge(df_merged, df_vaccine, on=['Country/Region', 'Date'], how='outer')
if 'Country/Region' in df_merged.columns:
      df_merged.loc[df_merged['Country/Region'] == 'US', 'Country/Region'] = 'United States'
      df_merged.loc[df_merged['Country/Region'] == 'Turkey', 'Country/Region'] = 'Turkiye'
      df_merged.loc[df_merged['Country/Region'] == 'Korea, South', 'Country/Region'] = 'South Korea'
      df_merged.loc[df_merged['Country/Region'] == 'Korea, North', 'Country/Region'] = 'North Korea'
      
    
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
df_combined = df_combined[~df_combined['Country/Region'].isin(['Winter Olympics 2022', 'Summer Olympics 2020'])]
df_combined = df_combined.drop(columns=['Year'])
df_combined = df_combined.dropna()
print(f"\nDate range in Combined DataFrame: {min_date} to {max_date}")
df_combined.to_csv(os.path.join(output_folder, "combined_dataset.csv"), index=False)

