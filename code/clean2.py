import pandas as pd
import os # Import os module to handle directory creation

# --- File Paths ---
# Assuming the script is run from the directory containing the 'data' folder
base_data_path = 'data' # Define base path for clarity

confirmed_file = os.path.join(base_data_path, 'raw/JHU/confirmed_global.csv')
deaths_file = os.path.join(base_data_path, 'raw/JHU/deaths_global.csv')
recovered_file = os.path.join(base_data_path, 'raw/JHU/recovered_global.csv') # Removed extra quotes
# Corrected vaccine file assignments:
vaccine_doses_file = os.path.join(base_data_path, 'raw/JHU/Vaccine/vaccine_doses_admin_global.csv') # Detailed doses file
vaccine_people_file = os.path.join(base_data_path, 'raw/JHU/Vaccine/vaccine_global.csv') # People/one dose file
economic_file = os.path.join(base_data_path, 'raw/WorldBank/economicIndicators.csv')

# --- Output Path ---
output_dir = os.path.join(base_data_path, 'merged')
output_file = os.path.join(output_dir, 'merged_covid_economic_data_v2.csv')

# Create the output directory if it doesn't exist
print(f"Ensuring output directory exists: {output_dir}")
os.makedirs(output_dir, exist_ok=True) # Creates directory if needed, doesn't error if it exists

# --- Load Data ---
print(f"Loading confirmed data from: {confirmed_file}")
df_confirmed = pd.read_csv(confirmed_file)
print(f"Loading deaths data from: {deaths_file}")
df_deaths = pd.read_csv(deaths_file)
print(f"Loading recovered data from: {recovered_file}")
df_recovered = pd.read_csv(recovered_file)
print(f"Loading vaccine doses data (expected wide format) from: {vaccine_doses_file}")
df_vaccine_doses = pd.read_csv(vaccine_doses_file)
print(f"Loading vaccine people data (expected long format) from: {vaccine_people_file}")
df_vaccine_people = pd.read_csv(vaccine_people_file)
print(f"Loading economic data from: {economic_file}")
df_econ = pd.read_csv(economic_file)
print("--- Data Loading Complete ---")

# --- Helper Function to Reshape JHU Data ---
def reshape_jhu_data(df, value_name):
    """Melts JHU data from wide to long format."""
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
    # Find date columns (adjust if format changes)
    # More robust date check: looks for MM/DD/YY format primarily
    date_cols = [col for col in df.columns if (isinstance(col, str) and len(col.split('/')) == 3 and col.split('/')[2].isdigit())]
    if not date_cols: # Fallback for other potential date formats like YYYY-MM-DD if first check fails
         date_cols = [col for col in df.columns if isinstance(col, str) and ( ('/' in col or '-' in col) and col[0].isdigit() )]

    print(f"Reshaping {value_name}: Found {len(date_cols)} date columns.")
    if not date_cols:
        print(f"Warning: No date columns found for {value_name}. Check file format.")
        # Return an empty DataFrame or handle error as appropriate
        return pd.DataFrame(columns=id_vars + ['Date', value_name, 'Region_Key'])


    df_long = pd.melt(df,
                      id_vars=id_vars,
                      value_vars=date_cols,
                      var_name='Date',
                      value_name=value_name)
    # Convert Date column to datetime objects, coercing errors to NaT (Not a Time)
    df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
    # Drop rows where Date conversion failed
    original_rows = len(df_long)
    df_long.dropna(subset=['Date'], inplace=True)
    if len(df_long) < original_rows:
        print(f"Dropped {original_rows - len(df_long)} rows due to invalid date formats during reshape for {value_name}.")

    # Handle potential missing Lat/Long if needed later for merging
    df_long[['Lat', 'Long']] = df_long[['Lat', 'Long']].fillna(0)
    # Standardize region identifier
    df_long['Province/State'] = df_long['Province/State'].fillna('Unknown') # Fill NaN states before creating key
    df_long['Region_Key'] = df_long['Country/Region'] + '_' + df_long['Province/State']
    return df_long

# --- Reshape Confirmed, Deaths, Recovered ---
print("--- Reshaping JHU Case/Death/Recovered Data ---")
df_confirmed_long = reshape_jhu_data(df_confirmed, 'Confirmed')
df_deaths_long = reshape_jhu_data(df_deaths, 'Deaths')
df_recovered_long = reshape_jhu_data(df_recovered, 'Recovered')

# --- Merge Confirmed, Deaths, Recovered ---
print("--- Merging JHU Case/Death/Recovered Data ---")
df_jhu_cases = pd.merge(df_confirmed_long, df_deaths_long[['Region_Key', 'Date', 'Deaths']], on=['Region_Key', 'Date'], how='left')
df_jhu_cases = pd.merge(df_jhu_cases, df_recovered_long[['Region_Key', 'Date', 'Recovered']], on=['Region_Key', 'Date'], how='left')
print(f"Merged JHU cases shape: {df_jhu_cases.shape}")

# --- Reshape Vaccine Doses Data (from vaccine_doses_admin_global.csv) ---
# This file is expected to be wide, similar to cases/deaths but with different ID columns
print("--- Reshaping JHU Vaccine Doses Data ---")
# Define expected ID columns for vaccine_doses_admin_global.csv
id_vars_vax_admin = ['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Province_State', 'Country_Region', 'Lat', 'Long_', 'Combined_Key', 'Population']
# Filter out columns that are not ID variables to find potential date columns
potential_date_cols_vax = [col for col in df_vaccine_doses.columns if col not in id_vars_vax_admin]
# Refine date column detection (example: check if it looks like a date)
date_cols_vax = [col for col in potential_date_cols_vax if isinstance(col, str) and ( ('/' in col or '-' in col) and col[0].isdigit() )]

print(f"Reshaping Vaccine Doses: Found {len(date_cols_vax)} date columns.")
if not date_cols_vax:
     print("Warning: No date columns found for Vaccine Doses. Check file format. Attempting to proceed without reshaping doses.")
     # Create an empty placeholder or handle based on actual file structure if it's already long
     df_vaccine_doses_long = pd.DataFrame() # Placeholder
else:
    # Ensure all expected ID columns exist, add missing ones if necessary
    for col in id_vars_vax_admin:
        if col not in df_vaccine_doses.columns:
            print(f"Warning: Column '{col}' not found in vaccine doses file. Adding it as NaN.")
            df_vaccine_doses[col] = pd.NA

    df_vaccine_doses_long = pd.melt(df_vaccine_doses,
                                    id_vars=id_vars_vax_admin,
                                    value_vars=date_cols_vax,
                                    var_name='Date_Str', # Keep as string initially
                                    value_name='Doses_Admin')
    df_vaccine_doses_long['Date'] = pd.to_datetime(df_vaccine_doses_long['Date_Str'], errors='coerce')
    original_rows = len(df_vaccine_doses_long)
    df_vaccine_doses_long.dropna(subset=['Date'], inplace=True)
    if len(df_vaccine_doses_long) < original_rows:
         print(f"Dropped {original_rows - len(df_vaccine_doses_long)} rows due to invalid date formats during reshape for Vaccine Doses.")
    # Create Region_Key for merging, handle missing Province_State
    df_vaccine_doses_long['Province_State'] = df_vaccine_doses_long['Province_State'].fillna('Unknown')
    df_vaccine_doses_long['Region_Key'] = df_vaccine_doses_long['Country_Region'] + '_' + df_vaccine_doses_long['Province_State']
    # Keep only necessary columns for merging
    df_vaccine_doses_long = df_vaccine_doses_long[['Region_Key', 'Date', 'Doses_Admin', 'Population']].copy()
    print(f"Reshaped vaccine doses shape: {df_vaccine_doses_long.shape}")


# --- Process vaccine_global.csv to get 'People_at_least_one_dose' ---
# This file is expected to be mostly long format already
print("--- Processing JHU Vaccine People Data ---")
# Check if expected columns exist
required_people_cols = ['Date', 'UID', 'Country_Region', 'Province_State', 'People_at_least_one_dose']
missing_cols = [col for col in required_people_cols if col not in df_vaccine_people.columns]
if missing_cols:
    print(f"Warning: Columns missing in vaccine_people file: {missing_cols}. Proceeding with available columns.")
    # Select only available columns from the required list
    available_people_cols = [col for col in required_people_cols if col in df_vaccine_people.columns]
    df_vaccine_people_proc = df_vaccine_people[available_people_cols].copy()
else:
    df_vaccine_people_proc = df_vaccine_people[required_people_cols].copy()

# Proceed only if essential columns ('Date', 'Country_Region', 'People_at_least_one_dose') are present
if 'Date' in df_vaccine_people_proc.columns and 'Country_Region' in df_vaccine_people_proc.columns and 'People_at_least_one_dose' in df_vaccine_people_proc.columns:
    df_vaccine_people_proc['Date'] = pd.to_datetime(df_vaccine_people_proc['Date'], errors='coerce')
    original_rows = len(df_vaccine_people_proc)
    df_vaccine_people_proc.dropna(subset=['Date'], inplace=True)
    if len(df_vaccine_people_proc) < original_rows:
         print(f"Dropped {original_rows - len(df_vaccine_people_proc)} rows due to invalid date formats during processing for Vaccine People.")

    # Handle Province_State if it exists, otherwise create key based only on Country
    if 'Province_State' in df_vaccine_people_proc.columns:
        df_vaccine_people_proc['Province_State'] = df_vaccine_people_proc['Province_State'].fillna('Unknown')
        df_vaccine_people_proc['Region_Key'] = df_vaccine_people_proc['Country_Region'] + '_' + df_vaccine_people_proc['Province_State']
    else:
        print("Warning: 'Province_State' not found in vaccine_people file. Creating Region_Key based on Country only.")
        df_vaccine_people_proc['Region_Key'] = df_vaccine_people_proc['Country_Region'] + '_Unknown' # Match format

    # Keep only necessary columns for merging
    cols_to_keep = ['Region_Key', 'Date', 'People_at_least_one_dose']
    df_vaccine_people_proc = df_vaccine_people_proc[[col for col in cols_to_keep if col in df_vaccine_people_proc.columns]].copy()
    print(f"Processed vaccine people shape: {df_vaccine_people_proc.shape}")
else:
    print("Error: Essential columns missing in vaccine_people file. Cannot process people vaccinated data.")
    df_vaccine_people_proc = pd.DataFrame() # Create empty df to avoid error on merge

# --- Merge JHU Cases with Vaccine Data ---
print("--- Merging JHU Cases with Vaccine Data ---")
df_merged = df_jhu_cases.copy() # Start with the cases data

# Merge doses data if it was successfully processed
if not df_vaccine_doses_long.empty:
    df_merged = pd.merge(df_merged, df_vaccine_doses_long, on=['Region_Key', 'Date'], how='left')
    print("Merged with vaccine doses data.")
else:
    print("Skipping merge with vaccine doses data (was not processed).")

# Merge people data if it was successfully processed
if not df_vaccine_people_proc.empty and 'People_at_least_one_dose' in df_vaccine_people_proc.columns:
     # Ensure the column exists before selecting it for merge
    cols_for_merge = ['Region_Key', 'Date', 'People_at_least_one_dose']
    df_merged = pd.merge(df_merged, df_vaccine_people_proc[cols_for_merge], on=['Region_Key', 'Date'], how='left')
    print("Merged with vaccine people data.")
else:
    print("Skipping merge with vaccine people data (was not processed or missing column).")

print(f"Shape after merging all JHU data: {df_merged.shape}")

# --- Process Economic Data ---
print("--- Processing Economic Data ---")
# Define exact series names to keep
econ_indicators = [
    'GDP growth (annual %)',
    'Inflation, consumer prices (annual %)',
    'Unemployment, total (% of total labor force) (national estimate)'
]
# Filter for the required indicators
df_econ_filtered = df_econ[df_econ['Series Name'].isin(econ_indicators)].copy()

# Melt the economic data
df_econ_long = pd.melt(df_econ_filtered,
                       id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
                       var_name='Year_Str',
                       value_name='Value')

# Extract year and convert to integer
df_econ_long['Year'] = df_econ_long['Year_Str'].str.extract(r'(\d{4})')
# Convert 'Value' to numeric, coercing errors to NaN
df_econ_long['Value'] = pd.to_numeric(df_econ_long['Value'], errors='coerce')
# Drop rows where year extraction or value conversion failed
original_rows = len(df_econ_long)
df_econ_long.dropna(subset=['Year', 'Value'], inplace=True)
print(f"Dropped {original_rows - len(df_econ_long)} rows from economic data due to missing year/value.")
df_econ_long['Year'] = df_econ_long['Year'].astype(int)


# Pivot the table to have indicators as columns
df_econ_pivot = df_econ_long.pivot_table(index=['Country Name', 'Year'],
                                         columns='Series Name',
                                         values='Value').reset_index()

# Rename columns for simplicity and merging
df_econ_pivot.rename(columns={
    'Country Name': 'Country/Region',
    'GDP growth (annual %)': 'GDP_Growth',
    'Inflation, consumer prices (annual %)': 'Inflation',
    'Unemployment, total (% of total labor force) (national estimate)': 'Unemployment'
}, inplace=True)

# Select only necessary columns
df_econ_final = df_econ_pivot[['Country/Region', 'Year', 'GDP_Growth', 'Inflation', 'Unemployment']].copy()
print(f"Processed economic data shape: {df_econ_final.shape}")

# --- Merge Economic Data with Daily Data ---
print("--- Merging Economic Data with Daily JHU Data ---")
df_merged['Year'] = df_merged['Date'].dt.year
# Before merging, check for consistency in Country/Region names if issues arise
# Example: df_merged['Country/Region'].unique() vs df_econ_final['Country/Region'].unique()
df_final = pd.merge(df_merged, df_econ_final, on=['Country/Region', 'Year'], how='left')
print(f"Final shape after merging economic data: {df_final.shape}")

# --- Final Cleaning (Example) ---
print("--- Performing Final Cleaning ---")
# Example: Fill missing numeric JHU data with 0 (consider alternatives like ffill or interpolation)
cols_to_fill_zero = ['Confirmed', 'Deaths', 'Recovered', 'Doses_Admin', 'People_at_least_one_dose']
for col in cols_to_fill_zero:
    if col in df_final.columns:
        print(f"Filling NaNs with 0 for column: {col}")
        df_final[col].fillna(0, inplace=True)
    else:
        print(f"Column {col} not found for filling NaNs.")


# Note: Missing economic indicators will remain NaN after the merge,
# as filling with 0 might be misleading for rates/percentages.

# Drop helper/intermediate columns if they exist
cols_to_drop = ['Year_Str', 'Date_Str', 'Region_Key']
df_final = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns], errors='ignore')
print("Dropped helper columns.")

print("\n--- Final Data Preview ---")
print("Preview of the merged dataset with Economic Indicators:")
print(df_final.head())
print("\nDataset Info:")
df_final.info()
print("\nCheck a few rows with merged economic data (Example: USA, 2021):")
print(df_final[(df_final['Year'] == 2021) & (df_final['Country/Region'] == 'US')].dropna(subset=['GDP_Growth', 'Inflation', 'Unemployment']).head())


# --- Save the Merged Dataset ---
print(f"\nAttempting to save merged data to: {output_file}")
try:
    df_final.to_csv(output_file, index=False)
    print(f"--- Merged data successfully saved to {output_file} ---")
except Exception as e:
    print(f"Error saving file: {e}")