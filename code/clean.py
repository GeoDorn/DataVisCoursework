import os
import pandas as pd

# This script is used to clean the data from the JHU and World Bank datasets.
# It loads the data, drops any rows with missing values, changes column headings to be more consistent, and converts the data from wide format to long format.

output_folder = "data/cleaned"

#Loads the data from the JHU and World Bank datasets and drops any rows with missing values
def load_data():
    # Confirmed_Global
    df1 = pd.read_csv("data/raw/JHU/confirmed_global.csv")
    #df1 = df1.dropna()
    print("Confirmed Cases loaded ")

    # Deaths_Global
    df2 = pd.read_csv("data/raw/JHU/deaths_global.csv")
    #df2 = df2.dropna()
    print("Deaths loaded")

    # Recovered_Global
    df3 = pd.read_csv("data/raw/JHU/recovered_global.csv")
    #df3 = df3.dropna()
    print("Recovered loaded")

    # Vaccine_Global
    df4 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_global.csv")
    #df4 = df4.dropna()
    print("Vaccines loaded")

    # Vaccine_doses_admin_global
    df5 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_doses_admin_global.csv")
    #df5 = df5.dropna()
    print("Doses Administered loaded")

    # EconomicIndicatorsData
    df6 = pd.read_csv("data/raw/WorldBank/economicIndicators.csv")
    #df6 = df6.dropna()
    print("Economic Indicators loaded")

    return df1, df2, df3, df4, df5, df6

def drop_columns(df1, df2, df3, df4, df5, df6):
    df1 = df1.drop(columns=['Lat', 'Long'])
    df2 = df2.drop(columns=['Lat', 'Long'])
    df3 = df3.drop(columns=['Lat', 'Long'])
    df4 = df4.drop(columns=['UID', ])
    df5 = df5.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Lat', 'Long_', 'Combined_Key'])
    df6 = df6.drop(columns=['Series Code', 'Country Code'])

    return df1, df2, df3, df4, df5, df6

def rename_columns(df4, df5, df6):
    
    df4 = df4.rename(columns={'Country_Region': 'Country/Region'})
    df5 = df5.rename(columns={'Country_Region': 'Country/Region'})
    df6 = df6.rename(columns={'Country Name': 'Country/Region'})

    return df4, df5, df6

def wide_to_long(df1, df2, df3, df4, df5, df6): 
    #
    df1 = pd.melt(df1, id_vars=['Country/Region'], var_name='Date', value_name='Confirmed')
    df2 = pd.melt(df2, id_vars=['Country/Region'], var_name='Date', value_name='Deaths')
    df3 = pd.melt(df3, id_vars=['Country/Region'], var_name='Date', value_name='Recovered')
    df4 = pd.melt(df4, id_vars=['Country/Region'], var_name='Date', value_name='Vaccines Administered')
    df5 = pd.melt(df5, id_vars=['Country/Region'], var_name='Date', value_name='Doses Administered')
    df6 = pd.melt(df6, id_vars=['Country/Region'], var_name='Date', value_name='Economic Indicators')

    return df1, df2, df3, df4, df5, df6

if __name__ == "__main__":
    # Load and clean data
    df1, df2, df3, df4, df5, df6 = load_data()
    #df4, df5, df6 = rename_columns(df4, df5, df6)
    #df1, df2, df3, df4, df5, df6 = wide_to_long(df1, df2, df3, df4, df5, df6)
    #df1, df2, df3, df4, df5, df6 = drop_columns(df1, df2, df3, df4, df5, df6)
   
    # Save cleaned data
    df1.to_csv(os.path.join(output_folder, "cleaned_confirmed_long.csv"), index=False)
    df2.to_csv(os.path.join(output_folder, "cleaned_deaths_long.csv"), index=False)   
    df3.to_csv(os.path.join(output_folder, "cleaned_recovered_long.csv"), index=False)
    df4.to_csv(os.path.join(output_folder, "cleaned_vaccine_long.csv"), index=False)
    df5.to_csv(os.path.join(output_folder, "cleaned_vaccine_doses_long.csv"), index=False)
    df6.to_csv(os.path.join(output_folder, "cleaned_economic_indicators_long.csv"), index=False)