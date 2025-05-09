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
    print(df1.head())
    print("Confirmed Cases loaded ")

    # Deaths_Global
    df2 = pd.read_csv("data/raw/JHU/deaths_global.csv")
    #df2 = df2.dropna()
    print(df2.head())
    print("Deaths loaded")

    # Recovered_Global
    df3 = pd.read_csv("data/raw/JHU/recovered_global.csv")
    #df3 = df3.dropna()
    print(df3.head())
    print("Recovered loaded")

    # Vaccine_Global
    df4 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_global.csv")
    #df4 = df4.dropna()
    print(df4.head())
    print("Vaccines loaded")

    # Vaccine_doses_admin_global
    df5 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_doses_admin_global.csv")
    #df5 = df5.dropna()
    #print(df5.head())
    #print("Doses Administered loaded")

    # EconomicIndicatorsData
    df6 = pd.read_csv("data/raw/WorldBank/economicIndicators.csv")
    #df6 = df6.dropna()
    print(df6.head())
    print("Economic Indicators loaded")

    return df1, df2, df3, df4, df5, df6

def clear_empty_rows(df1, df2, df3, df4, df5, df6):
    df1 = df1.fillna(0)
    print("Confirmed Cases cleaned")
    df2 = df2.fillna(0)
    print("Deaths cleaned")
    df3 = df3.fillna(0)
    print("Recovered cleaned")
    #df4 = df4.fillna(0)
    print("Vaccines cleaned")
    #df5 = df5.fillna(0)
    print("Doses Administered cleaned")
    #df6 = df6.dropna()
    print("Economic Indicators cleaned")
    return df1, df2, df3, df4, df5, df6

def drop_columns(df1, df2, df3, df4, df5, df6):
    df1 = df1.drop(columns=['Province/State', 'Lat', 'Long'])
    df2 = df2.drop(columns=['Province/State', 'Lat', 'Long'])
    df3 = df3.drop(columns=['Province/State', 'Lat', 'Long'])
    df4 = df4.drop(columns=['UID', 'Province_State', 'Doses_admin'])
    df5 = df5.drop(columns=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2', 'Lat', 'Long_', 'Combined_Key', 'Province_State'])
    df6 = df6.drop(columns=['Series Code', 'Country Code'])

    return df1, df2, df3, df4, df5, df6

def rename_columns(df4, df5, df6):
    
    df4 = df4.rename(columns={'Country_Region': 'Country/Region'})
    df5 = df5.rename(columns={'Country_Region': 'Country/Region'})
    df6 = df6.rename(columns={'Country Name': 'Country/Region','2020 [YR2020]': 'Year_2020', '2021 [YR2021]': 'Year_2021', '2022 [YR2022]': 'Year_2022', '2023 [YR2023]': 'Year_2023'})
    
    return df4, df5, df6

def rename_country_format(df1, df2, df3, df4, df5, df6):
    country_replacements = {
        'Venezuela, RB': 'Venezuela',
        'Korea, Rep.': 'South Korea',
        'Korea, South': 'South Korea',
        'Korea, North': 'North Korea',
        'United States': 'United States',
        'US': 'United States',
        'Viet Nam': 'Vietnam',
        'Russian Federation': 'Russia',
        'Egypt, Arab Rep.': 'Egypt',
        'Iran, Islamic Rep.': 'Iran',
        'Bahamas, The': 'Bahamas',
        'Brunei Darussalam': 'Brunei',
        'Congo, Rep.': 'Congo (Brazzaville)',
        'Congo, Dem. Rep.': 'Congo (Kinshasa)',
        'Gambia, The': 'Gambia',
        'Lao PDR': 'Laos',
        'Micronesia, Fed. Sts.': 'Micronesia',
        'Slovak Republic': 'Slovakia',
        'Syrian Arab Republic': 'Syria',
        'Turkiye': 'Turkey',
        'Saint Kitts and Nevis': 'St. Kitts and Nevis',
        'Saint Lucia': 'St. Lucia',
        'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines'
    }

    for df in [df1, df2, df3, df4, df5, df6]:
        if 'Country/Region' in df.columns:
            df['Country/Region'] = df['Country/Region'].replace(country_replacements)

    return df1, df2, df3, df4, df5, df6

def group_by_country(df1, df2, df3, df4, df5, df6):
    df1 = df1.groupby(['Country/Region']).sum().reset_index()
    df2 = df2.groupby(['Country/Region']).sum().reset_index()
    df3 = df3.groupby(['Country/Region']).sum().reset_index()
    #df4 = df4.groupby(['Country/Region']).sum().reset_index()
    df5 = df5.groupby(['Country/Region']).sum().reset_index()
    #df6 = df6.groupby(['Country/Region']).sum().reset_index()
    return df1, df2, df3, df4, df5, df6


def wide_to_long(df1, df2, df3, df4, df5, df6): 
    
    df1 = pd.melt(df1, id_vars=['Country/Region'], var_name='Date', value_name='Confirmed')
    df2 = pd.melt(df2, id_vars=['Country/Region'], var_name='Date', value_name='Deaths')
    df3 = pd.melt(df3, id_vars=['Country/Region'], var_name='Date', value_name='Recovered')
    df5 = pd.melt(df5, id_vars=['Country/Region', 'Population'], var_name='Date', value_name='Doses Administered')
    df6 = pd.melt(df6, id_vars=['Series Name', 'Country/Region'],
                  value_vars=['Year_2020', 'Year_2021', 'Year_2022', 'Year_2023'],
                  var_name='Year',
                  value_name='Value')

    return df1, df2, df3, df4, df5, df6

def convert_date(df1, df2, df3, df4, df6):
    # Convert the date column to datetime format
    df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%y', errors='coerce')
    df2['Date'] = pd.to_datetime(df2['Date'], format='%m/%d/%y', errors='coerce')
    df3['Date'] = pd.to_datetime(df3['Date'], format='%m/%d/%y', errors='coerce')
    #df4['Date'] = pd.to_datetime(df4['Date'], format='%m/%d/%y', errors='coerce')
    df6['Year'] = df6['Year'].str.replace('Year_', '', regex=False).astype(int)
    return df1, df2, df3, df4, df6

if __name__ == "__main__":
    # Load and clean data
    df1, df2, df3, df4, df5, df6 = load_data()
    df1, df2, df3, df4, df5, df6 = clear_empty_rows(df1, df2, df3, df4, df5, df6)
    df4, df5, df6 = rename_columns(df4, df5, df6)
    df1, df2, df3, df4, df5, df6 = drop_columns(df1, df2, df3, df4, df5, df6)
    df1, df2, df3, df4, df5, df6 = rename_country_format(df1, df2, df3, df4, df5, df6)
    df1, df2, df3, df4, df5, df6 = group_by_country(df1, df2, df3, df4, df5, df6)
    df1, df2, df3, df4, df5, df6 = wide_to_long(df1, df2, df3, df4, df5, df6)
    df1, df2, df3, df4, df6 = convert_date(df1, df2, df3, df4, df6)
    df6['Value'] = df6['Value'].replace('..', pd.NA)
    df6['Value'] = pd.to_numeric(df6['Value'], errors='coerce')

    # Save cleaned data
    df1.to_csv(os.path.join(output_folder, "cleaned_confirmed_cases.csv"), index=False)
    df2.to_csv(os.path.join(output_folder, "cleaned_deaths_long.csv"), index=False)   
    df3.to_csv(os.path.join(output_folder, "cleaned_recovered_long.csv"), index=False)
    df4.to_csv(os.path.join(output_folder, "cleaned_vaccine_long.csv"), index=False)
    df5.to_csv(os.path.join(output_folder, "cleaned_vaccine_doses_long.csv"), index=False)
    df6.to_csv(os.path.join(output_folder, "cleaned_economic_indicators_long.csv"), index=False)