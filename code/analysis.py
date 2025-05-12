import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define visualizations folder and create if it doesn't exist
visualizations_folder = "visualizations"
os.makedirs(visualizations_folder, exist_ok=True)

# ===  Load and Preprocess Data ===
# Read the merged dataset
df = pd.read_csv('data/merged/combined_dataset.csv')

# Feature engineering
df['Vaccination Rate'] = df['People_at_least_one_dose'] / df['Population, total'] * 100
df['Cases per 100k'] = df['Confirmed'] / df['Population, total'] * 100000
df['Deaths per 100k'] = df['Deaths'] / df['Population, total'] * 100000

# Save cleaned dataset
final_path = os.path.join("data/complete/final_dataset.csv")
df.to_csv(final_path, index=False)

# Reload the cleaned dataset
df = pd.read_csv(final_path)

# Drop rows with missing critical data
df = df.dropna(subset=["Country/Region", "Vaccination Rate", "GDP growth (annual %)"])

# ===  KMeans Clustering ===
df_latest = df.sort_values("Date").groupby("Country/Region").last().reset_index()
features = df_latest[["Country/Region", "Vaccination Rate", "GDP growth (annual %)"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[["Vaccination Rate", "GDP growth (annual %)"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto') 
features["Cluster"] = kmeans.fit_predict(X_scaled)

clustered = features[["Country/Region", "Vaccination Rate", "GDP growth (annual %)", "Cluster"]]
clustered.to_csv("clustered_countries.csv", index=False) 

sil_score = silhouette_score(X_scaled, features["Cluster"])
print(f"Silhouette Score: {sil_score}")

inertia = kmeans.inertia_
print(f"Inertia: {inertia}")

cluster_summary = clustered.groupby("Cluster").agg({
    "Vaccination Rate": "mean",
    "GDP growth (annual %)": "mean",
    "Country/Region": "count"
}).rename(columns={
    "Vaccination Rate": "Avg Vaccination Rate",
    "GDP growth (annual %)": "Avg GDP Growth",
    "Country/Region": "Countries in Cluster"
})
print(cluster_summary)
# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustered, x="Vaccination Rate", y="GDP growth (annual %)", hue="Cluster", palette="Set2", s=100)
plt.title("Country Clustering by Vaccination Rate and GDP Growth")
plt.xlabel("Vaccination Rate (%)")
plt.ylabel("GDP Growth (annual %)")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "country_clustering_vaccination_gdp.png"))


# ===  Linear Regression ===
X = df_latest[["Vaccination Rate"]]
y = df_latest["GDP growth (annual %)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

# Plot regression
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Linear Regression: Vaccination Rate vs. GDP Growth")
plt.xlabel("Vaccination Rate (%)")
plt.ylabel("GDP Growth (annual %)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "linear_regression_vaccination_gdp.png"))


# ===  Correlation Heatmap ===
cols_to_convert = [
    "Confirmed", "Deaths", "People_at_least_one_dose", "Doses Administered",
    "GDP growth (annual %)", "Inflation, consumer prices (annual %)",
    "Population, total", "Unemployment, total (% of total labor force) (national estimate)",
    "Vaccination Rate", "Cases per 100k", "Deaths per 100k"
]

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

df_numeric = df[cols_to_convert].dropna()
correlation_matrix = df_numeric.corr()

print("\nCorrelation Matrix:\n")
print(correlation_matrix.round(2))

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "correlation_matrix.png"))


# === Plotly Choropleth map of Vaccination Rate ===
#latest_cases = df.sort_values('Date').drop_duplicates('Country/Region', keep='last')
#fig_choropleth = px.choropleth(latest_cases,
#                               locations="Country/Region",
#                               locationmode="country names",
#                               color="Vaccination Rate",
#                               color_continuous_scale="Blues",
#                               title="Global Vaccination Rate (%)")
#fig_choropleth.show()

# === Early Rollout = Less deaths ===
df["Date"] = pd.to_datetime(df["Date"])
rollout_dates = df[df['People_at_least_one_dose'].notna()] \
    .groupby('Country/Region')['Date'].min().reset_index()
rollout_dates.columns = ['Country/Region', 'Vaccine_Rollout_Date']

# 2.2 Find the latest available Deaths per 100k for each country
latest_deaths = df.sort_values('Date') \
    .groupby('Country/Region').tail(1)[['Country/Region', 'Deaths per 100k']]

# 2.3 Merge the two dataframes
analysis_df = rollout_dates.merge(latest_deaths, on='Country/Region')

# 2.4 Convert rollout date to numeric (days since Jan 1, 2020)
base_date = pd.to_datetime('2020-01-01')
analysis_df['Days_To_Rollout'] = (analysis_df['Vaccine_Rollout_Date'] - base_date).dt.days

# 2.5 Plot Rollout Time vs Deaths
plt.figure(figsize=(10, 6))
sns.regplot(data=analysis_df, x='Days_To_Rollout', y='Deaths per 100k')
plt.title("Earlier Vaccine Rollouts Correlate with Lower Deaths")
plt.xlabel("Days Until First Vaccine Dose")
plt.ylabel("Deaths per 100k")
correlation = analysis_df['Days_To_Rollout'].corr(analysis_df['Deaths per 100k'])
print(f"Correlation between rollout timing and deaths per 100k: {correlation:.2f}")
plt.text(
    0.75, 0.95,
    f'Correlation: {correlation:.2f}',
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
)
plt.tight_layout()

# Save the plot

plt_path = os.path.join(visualizations_folder, "rollout_vs_deaths.png")
plt.savefig(plt_path)

print(f"\nAll visualizations have been attempted to be saved in the '{visualizations_folder}' folder.")