import pandas as pd
import glob
import os

output_folder = "data/collated"

def clean_data(): 
    #Confirmed_Global
    df1 = pd.read_csv("data/raw/JHU/confirmed_global.csv")
    df1 = df1.dropna()
    print("DF1 loaded ")
    #Deaths_Global
    df2 = pd.read_csv("data/raw/JHU/deaths_global.csv")
    df2 = df2.dropna()
    print("DF2 Loaded")
    #Recovered_Global
    df3 = pd.read_csv("data/raw/JHU/recovered_global.csv")
    df3 = df3.dropna()
    print("DF3 Loaded")
    #Vaccine_Global
    df4 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_global.csv")
    df4 = df4.dropna()
    print("DF4 Loaded")
    #Vaccine_doses_admin_global
    df5 = pd.read_csv("data/raw/JHU/Vaccine/vaccine_doses_admin_global.csv")
    df5 = df5.dropna()
    print("DF5 Loaded")
    #EconomicIndicatorsData
    df6 = pd.read_csv("data/raw/WorldBank/economicIndicatorsData.csv")
    df6 = df6.dropna()
    print("DF6 Loaded")
    return df1, df2, df3, df4, df5, df6

JHU_files = glob.glob("data/raw/JHU/*.csv")
vaccine_files= glob.glob("data/raw/JHU/Vaccine/*.csv")
World_Bank= glob.glob("data/raw/WorldBank/*.csv")



# Combine JHU files
JHU_dfs = [pd.read_csv(file) for file in JHU_files]
#JHU_combined = pd.concat(JHU_dfs, ignore_index=True)

# Combine Vaccine files
vaccine_dfs = [pd.read_csv(file) for file in vaccine_files]
#vaccine_combined = pd.concat(vaccine_dfs, ignore_index=True)

# Combine World Bank files
wb_dfs = [pd.read_csv(file) for file in World_Bank]
#wb_combined = pd.concat(wb_dfs, ignore_index=True)

#print(JHU_dfs.file[['Country/Region']].duplicated().sum())
#print(vaccine_combined[['Country_Region']].duplicated().sum())
#print(wb_combined[['Country']].duplicated().sum())
#print(JHU_combined.columns)
#print(vaccine_combined.columns)
#print(wb_combined.columns)

#JHU_combined.to_csv(os.path.join(output_folder, "JHU_combined.csv"), index=False)
#vaccine_combined.to_csv(os.path.join(output_folder, "vaccine_combined.csv"), index=False)
#wb_combined.to_csv(os.path.join(output_folder, "world_bank_combined.csv"), index=False)