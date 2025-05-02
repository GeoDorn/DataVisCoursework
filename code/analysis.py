# %% Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import necessary scikit-learn modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split # Optional for splitting data
from sklearn.metrics import mean_squared_error, r2_score
# Optional: from sklearn.preprocessing import StandardScaler

# %% Configuration
# Define the path to your cleaned data
cleaned_folder = "data/cleaned"

# Define the year for the regression analysis
ANALYSIS_YEAR_FOR_REGRESSION = 2021

# %% Load Data
print("--- Loading Data ---")
try:
    df_confirmed = pd.read_csv(f"{cleaned_folder}/cleaned_confirmed_cases.csv", parse_dates=['Date'])
    df_deaths = pd.read_csv(f"{cleaned_folder}/cleaned_deaths_long.csv", parse_dates=['Date'])
    df_recovered = pd.read_csv(f"{cleaned_folder}/cleaned_recovered_long.csv", parse_dates=['Date'])
    df_econ = pd.read_csv(f"{cleaned_folder}/cleaned_economic_indicators_long.csv")
    # Uncomment if you want to include vaccine data
    # df_vaccine = pd.read_csv(f"{cleaned_folder}/cleaned_vaccine_long.csv")

    print("Cleaned datasets loaded successfully.")
    print("\nEconomic Indicators Info:")
    df_econ.info()
    print("\nConfirmed Cases Info:")
    df_confirmed.info()

    data_loaded_successfully = True
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure the cleaned files are in the '{cleaned_folder}' directory.")
    data_loaded_successfully = False
except Exception as e:
    print(f"An error occurred during loading: {e}")
    data_loaded_successfully = False

# %% Prepare COVID Data for Yearly Merge (only if data loaded)
if data_loaded_successfully:
    print("\n--- Preparing COVID Data ---")
    if 'df_confirmed' in locals():
        df_confirmed['Year'] = df_confirmed['Date'].dt.year
        df_deaths['Year'] = df_deaths['Date'].dt.year
        df_recovered['Year'] = df_recovered['Date'].dt.year

        # Get the maximum cumulative value for each country/year (value at end of year)
        # Using idxmax() finds the index of the max date, then .loc retrieves that row
        print("Aggregating COVID data to yearly...")
        df_confirmed_yearly = df_confirmed.loc[df_confirmed.groupby(['Country/Region', 'Year'])['Date'].idxmax()]
        df_deaths_yearly = df_deaths.loc[df_deaths.groupby(['Country/Region', 'Year'])['Date'].idxmax()]
        df_recovered_yearly = df_recovered.loc[df_recovered.groupby(['Country/Region', 'Year'])['Date'].idxmax()]

        # Select and rename columns for clarity before merging
        df_confirmed_yearly = df_confirmed_yearly[['Country/Region', 'Year', 'Confirmed']].rename(columns={'Confirmed': 'Confirmed_EndOfYear'})
        df_deaths_yearly = df_deaths_yearly[['Country/Region', 'Year', 'Deaths']].rename(columns={'Deaths': 'Deaths_EndOfYear'})
        df_recovered_yearly = df_recovered_yearly[['Country/Region', 'Year', 'Recovered']].rename(columns={'Recovered': 'Recovered_EndOfYear'})

        print("COVID data aggregated yearly (using end-of-year cumulative values).")
        covid_data_prepared = True
    else:
        print("Confirmed cases DataFrame not found. Cannot prepare COVID data.")
        covid_data_prepared = False
else:
    covid_data_prepared = False

# %% Prepare Economic Data (Pivot)
if data_loaded_successfully:
    print("\n--- Preparing Economic Data ---")
    # Pivot the economic indicators data to have indicators as columns
    print("Pivoting Economic Indicators data...")
    try:
        df_econ_pivot = df_econ.pivot_table(index=['Country/Region', 'Year'],
                                            columns='Series Name',
                                            values='Value').reset_index()
        # Clean up column names potentially introduced by pivot
        df_econ_pivot.columns.name = None
        print("Pivoted Economic Indicators data successfully.")
        print(f"Pivoted Econ data shape: {df_econ_pivot.shape}")
        econ_data_prepared = True
    except Exception as e:
        print(f"Error pivoting economic data: {e}")
        econ_data_prepared = False
else:
    econ_data_prepared = False


# %% Investigate and Standardize Country Names (CRITICAL STEP)
if covid_data_prepared and econ_data_prepared:
    print("\n--- Investigating Country Name Consistency ---")

    # Combine COVID data first before comparing with Econ data
    df_merged_covid = pd.merge(df_confirmed_yearly, df_deaths_yearly, on=['Country/Region', 'Year'], how='outer')
    df_merged_covid = pd.merge(df_merged_covid, df_recovered_yearly, on=['Country/Region', 'Year'], how='outer')

    unique_covid_countries = set(df_merged_covid['Country/Region'].unique())
    unique_econ_countries = set(df_econ_pivot['Country/Region'].unique())

    print(f"Unique countries in COVID data: {len(unique_covid_countries)}")
    print(f"Unique countries in Econ data: {len(unique_econ_countries)}")

    # Find names that don't match
    covid_only = sorted(list(unique_covid_countries - unique_econ_countries))
    econ_only = sorted(list(unique_econ_countries - unique_covid_countries))

    if not covid_only and not econ_only:
        print("\nCountry names appear consistent between COVID and Economic datasets.")
    else:
        print("\nPotential Country Name Mismatches Found!")
        if covid_only:
            print("\nCountries in COVID data BUT NOT Econ data:")
            print(covid_only)
        if econ_only:
            print("\nCountries in Econ data BUT NOT COVID data:")
            print(econ_only)

        # --- Define Country Name Mapping (USER MUST EDIT THIS) ---
        # Create a dictionary to map names from one standard to another
        # Example: {'JHU Name': 'World Bank Name', ...}
        # Fill this based on the differences printed above!
        country_mapping = {
            # --- Example Mappings - Replace with your actual findings ---
             'Burma': 'Myanmar',
             'Congo (Brazzaville)': 'Congo, Rep.',
             'Congo (Kinshasa)': 'Congo, Dem. Rep.',
             "Cote d'Ivoire": "Cote d'Ivoire", # Check if case/apostrophe matches
             'Czechia': 'Czech Republic', # Or vice-versa depending on your preferred standard
             'Korea, South': 'Korea, Rep.',
             'Saint Kitts and Nevis': 'St. Kitts and Nevis',
             'Saint Lucia': 'St. Lucia',
             'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
             'Taiwan*': 'Taiwan', # Example if one has an asterisk
             'US': 'United States',
             'West Bank and Gaza': 'Palestine', # Example, need to check actual names used
             # Add all other necessary mappings based on the print output above
             # Ensure you map TO a consistent set of names present in ONE of the datasets,
             # or map BOTH to a new standard name.
        }
        print("\nApplying country name standardization mapping...")
        # Apply the mapping to BOTH dataframes before merging
        # Make sure the keys in your mapping match names in the df being modified
        df_merged_covid['Country/Region'] = df_merged_covid['Country/Region'].replace(country_mapping)
        df_econ_pivot['Country/Region'] = df_econ_pivot['Country/Region'].replace(country_mapping)
        print("Standardization applied.")

# %% Merge Final DataFrames
if covid_data_prepared and econ_data_prepared:
    print("\n--- Merging COVID and Economic Data ---")
    # Merge standardized COVID data with standardized pivoted Economic data
    df_final = pd.merge(df_merged_covid, df_econ_pivot, on=['Country/Region', 'Year'], how='outer')

    print("\nMerged COVID and Economic data.")
    print(f"Final merged DataFrame shape: {df_final.shape}")
    print("Final merged DataFrame columns:", df_final.columns.tolist())
    print("Final merged DataFrame head:\n", df_final.head())
    merge_successful = True
else:
    print("\nSkipping final merge as data preparation failed.")
    merge_successful = False
    # Define df_final using only econ data if that's all that's available
    if econ_data_prepared:
         df_final = df_econ_pivot
         print("\nUsing pivoted Economic data only for analysis.")
         merge_successful = True # Allow EDA/ML on econ data only

# %% Exploratory Data Analysis (EDA)
if merge_successful and 'df_final' in locals():
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print("\nDataFrame Info:")
    df_final.info()

    print("\nMissing Values per Column (Top 20):")
    # Sort missing values to see the worst offenders first
    print(df_final.isnull().sum().sort_values(ascending=False).head(20))

    print("\nDescriptive Statistics (Numeric Columns):")
    # Ensure numeric_cols is defined before use
    numeric_cols_df = df_final.select_dtypes(include='number')
    if not numeric_cols_df.empty:
        print(numeric_cols_df.describe())
    else:
        print("No numeric columns found for descriptive statistics.")

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    # Plot GDP Growth Distribution
    if 'GDP growth (annual %)' in df_final.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_final['GDP growth (annual %)'].dropna(), kde=True)
        plt.title('Distribution of GDP Growth (Annual %)')
        plt.xlabel('GDP Growth (%)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    else:
        print("GDP growth column not found for plotting.")

    # Plot Average Inflation Trend
    if 'Inflation, consumer prices (annual %)' in df_final.columns:
        plt.figure(figsize=(10, 6))
        # Calculate mean inflation per year, handling potential non-numeric errors
        try:
            mean_inflation = df_final.groupby('Year')['Inflation, consumer prices (annual %)'].mean(numeric_only=True)
            if not mean_inflation.empty:
                 mean_inflation.plot(kind='line', marker='o')
                 plt.title('Average Global Inflation Over Years')
                 plt.xlabel('Year')
                 plt.ylabel('Average Inflation (%)')
                 plt.grid(True)
                 plt.tight_layout()
                 plt.show()
            else:
                 print("Could not calculate mean inflation (maybe no data?).")
        except TypeError as e:
             print(f"Could not plot inflation trend due to data type error: {e}")

    else:
        print("Inflation column not found for plotting.")

    # Plot Correlation Heatmap
    corr_cols = ['Confirmed_EndOfYear', 'Deaths_EndOfYear', 'Recovered_EndOfYear',
                 'GDP growth (annual %)', 'Inflation, consumer prices (annual %)',
                 'Unemployment, total (% of total labor force) (national estimate)']
    # Filter columns that actually exist in df_final and are numeric
    existing_corr_cols = [col for col in corr_cols if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col])]

    if len(existing_corr_cols) > 1:
        print(f"\nGenerating Correlation Heatmap for: {existing_corr_cols}")
        # Calculate correlation matrix, dropping rows with any NaNs in the selected columns for calculation
        correlation_matrix = df_final[existing_corr_cols].dropna().corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Selected Variables (Rows with any NaN dropped)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    else:
        print("\nNot enough numeric columns found/available for correlation heatmap.")

# %% Scikit-learn Analysis Example: Linear Regression
if merge_successful and 'df_final' in locals():
    print("\n--- Scikit-learn Analysis: Linear Regression ---")

    # Define target and feature variables
    target_variable = 'GDP growth (annual %)'
    feature_variable = 'Deaths_EndOfYear' # Using a single feature for simplicity

    # Check if necessary columns exist AFTER merge and standardization
    if target_variable in df_final.columns and feature_variable in df_final.columns:
        print(f"\nAttempting Linear Regression for {ANALYSIS_YEAR_FOR_REGRESSION}: Predicting '{target_variable}' using '{feature_variable}'.")

        # Filter data for the specific year
        df_year = df_final[df_final['Year'] == ANALYSIS_YEAR_FOR_REGRESSION].copy()

        # --- Preprocessing for the model ---
        # 1. Handle missing values specifically for the columns needed for regression
        required_cols = [feature_variable, target_variable]
        rows_before = len(df_year)
        df_model_data = df_year[required_cols].dropna()
        rows_after = len(df_model_data)

        if rows_before > 0 : # Check if there was any data for the year initially
             print(f"Started with {rows_before} rows for {ANALYSIS_YEAR_FOR_REGRESSION}.")
             print(f"Using {rows_after} rows after dropping missing values in '{feature_variable}' or '{target_variable}'.")
        else:
             print(f"No data found for the year {ANALYSIS_YEAR_FOR_REGRESSION}.")


        if rows_after > 1: # Need at least 2 data points for regression
            # 2. Define X (features) and y (target)
            X = df_model_data[[feature_variable]] # Keep X as a DataFrame (2D)
            y = df_model_data[target_variable]   # y is a Series (1D)

            # --- Train the Linear Regression Model ---
            model = LinearRegression()
            model.fit(X, y)

            # --- Inspect the Model ---
            print(f"\nModel Trained for {ANALYSIS_YEAR_FOR_REGRESSION}:")
            print(f"Intercept: {model.intercept_:.4f}")
            print(f"Coefficient for {feature_variable}: {model.coef_[0]:.8f}") # Access first element for single feature

            # --- Evaluate the Model ---
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            print(f"R-squared (on training data): {r2:.4f}")
            print(f"Mean Squared Error (on training data): {mse:.4f}")

            # --- Visualize the Regression Line ---
            plt.figure(figsize=(10, 6))
            # Use the original columns from df_model_data for plotting context
            sns.scatterplot(x=df_model_data[feature_variable], y=df_model_data[target_variable], alpha=0.7)
            # Plot regression line using the same data range
            plt.plot(df_model_data[feature_variable], y_pred, color='red', linewidth=2)
            plt.title(f'Linear Regression: {target_variable} vs {feature_variable} ({ANALYSIS_YEAR_FOR_REGRESSION})')
            plt.xlabel(feature_variable)
            plt.ylabel(target_variable)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        else:
            print(f"Not enough data points ({rows_after}) after dropping NaNs for {ANALYSIS_YEAR_FOR_REGRESSION} to perform regression.")

    else:
        # Print which specific column is missing
        missing_cols_msg = []
        if target_variable not in df_final.columns:
            missing_cols_msg.append(f"'{target_variable}'")
        if feature_variable not in df_final.columns:
            missing_cols_msg.append(f"'{feature_variable}'")
        print(f"Required column(s) {', '.join(missing_cols_msg)} not found in the final DataFrame. Skipping regression.")

print("\n--- Analysis Script Finished ---")