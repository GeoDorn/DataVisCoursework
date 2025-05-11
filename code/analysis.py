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

# === 1. Load and Preprocess Data ===
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

# === 2. KMeans Clustering ===
df_latest = df.sort_values("Date").groupby("Country/Region").last().reset_index()
features = df_latest[["Country/Region", "Vaccination Rate", "GDP growth (annual %)"]].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[["Vaccination Rate", "GDP growth (annual %)"]])

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto') # Added n_init for KMeans
features["Cluster"] = kmeans.fit_predict(X_scaled)

clustered = features[["Country/Region", "Vaccination Rate", "GDP growth (annual %)", "Cluster"]]
clustered.to_csv("clustered_countries.csv", index=False) # Keeps data CSV in root

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustered, x="Vaccination Rate", y="GDP growth (annual %)", hue="Cluster", palette="Set2", s=100)
plt.title("Country Clustering by Vaccination Rate and GDP Growth")
plt.xlabel("Vaccination Rate (%)")
plt.ylabel("GDP Growth (annual %)")
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
sil_score = silhouette_score(X_scaled, features["Cluster"])
print(f"Silhouette Score: {sil_score}")
plt.savefig(os.path.join(visualizations_folder, "country_clustering_vaccination_gdp.png"))
plt.show()

# === 3. Linear Regression ===
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
plt.show()

# === 4. Correlation Heatmap ===
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
plt.savefig(os.path.join(visualizations_folder, "correlation_heatmap.png"))
plt.show()

# === 5. Heatmap of Cases per 100k ===
latest_cases = df.sort_values('Date').drop_duplicates('Country/Region', keep='last')
top_countries = latest_cases.nlargest(30, 'Cases per 100k')['Country/Region']
heatmap_data_df = latest_cases[latest_cases['Country/Region'].isin(top_countries)].set_index('Country/Region')[['Cases per 100k']] # Ensure it's a DataFrame

# Using pivot_table is good if you have multiple entries per country and need aggregation.
# If set_index already gives unique countries, this is fine.
heatmap_matrix = heatmap_data_df.pivot_table(index='Country/Region', values='Cases per 100k', aggfunc='mean')


plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_matrix, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5)
plt.title('COVID-19 Cases per 100,000 People by Country', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "cases_per_100k_heatmap.png"))
plt.show()

# === 6. GDP Growth Trends by Country (Annual) ===
df['Year'] = pd.to_datetime(df['Date']).dt.year
country_gdp = df.groupby(['Country/Region', 'Year'])['GDP growth (annual %)'].mean().reset_index()

countries_to_plot = ['United States', 'China', 'Germany', 'Japan', 'United Kingdom',
                     'Brazil', 'India', 'South Africa', 'Australia']

plt.figure(figsize=(14, 8))
for country in countries_to_plot:
    if country in country_gdp['Country/Region'].values:
        subset = country_gdp[country_gdp['Country/Region'] == country]
        plt.plot(subset['Year'], subset['GDP growth (annual %)'], marker='o', linewidth=2, label=country)

plt.title('GDP Growth Trends by Country (Annual)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('GDP Growth (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "gdp_growth_trends_annual.png"))
plt.show()

# === 7. GDP vs Vaccination Rate Bar Chart ===
countries_to_compare = ['United States', 'China', 'Germany', 'Brazil', 'India',
                        'South Africa', 'United Kingdom', 'Israel', 'Japan', 'Australia']

# Ensure latest_cases is used here as intended by the original script context
comparison_data = latest_cases[latest_cases['Country/Region'].isin(countries_to_compare)].copy()
# Ensure the countries_to_compare are actually present in comparison_data and in the desired order for plotting
# Reindex comparison_data to match the order of countries_to_compare if necessary, and handle missing countries
comparison_data = comparison_data.set_index('Country/Region').reindex(countries_to_compare).reset_index()
comparison_data = comparison_data.dropna(subset=['GDP growth (annual %)', 'Vaccination Rate']) # Drop if key data is missing for a country

fig, ax1 = plt.subplots(figsize=(14, 8))
x = np.arange(len(comparison_data['Country/Region'])) # Use length of actual data to plot
width = 0.35

# Bar for GDP
ax1.bar(x - width/2, comparison_data['GDP growth (annual %)'], width, label='GDP Growth (%)', color='steelblue')
ax1.set_ylabel('GDP Growth (%)')
# Dynamic Y-limits for GDP
gdp_min = comparison_data['GDP growth (annual %)'].min()
gdp_max = comparison_data['GDP growth (annual %)'].max()
ax1.set_ylim([min(0, gdp_min) -1 , gdp_max + 1])


# Bar for Vaccination Rate
ax2 = ax1.twinx()
ax2.bar(x + width/2, comparison_data['Vaccination Rate'], width, label='Vaccination Rate (%)', color='orangered')
ax2.set_ylabel('Vaccination Rate (%)')
ax2.set_ylim([0, 105]) # Set Y-limit for percentage up to 100 (or slightly more for padding)

ax1.set_xticks(x)
ax1.set_xticklabels(comparison_data['Country/Region'], rotation=45, ha='right')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Vaccination Rates vs. GDP Growth by Country', fontsize=16)
plt.tight_layout()
fig.savefig(os.path.join(visualizations_folder, "gdp_vs_vaccination_barchart.png"))
plt.show()

# === 8. Plotly Choropleth map of Vaccination Rate ===
fig_choropleth = px.choropleth(latest_cases,
                               locations="Country/Region",
                               locationmode="country names",
                               color="Vaccination Rate",
                               color_continuous_scale="Blues",
                               title="Global Vaccination Rate (%)")
try:
    fig_choropleth.write_image(os.path.join(visualizations_folder, "global_vaccination_rate_map.png"))
    print(f"Successfully saved: global_vaccination_rate_map.png")
except Exception as e:
    print(f"Could not save Plotly figure as image. Ensure Kaleido is installed (pip install kaleido). Error: {e}")
fig_choropleth.show()


# === 9. Vaccination Rate Over Time for selected countries ===
# countries_to_plot is defined in section 6
vaccination_trend = df[df['Country/Region'].isin(countries_to_plot)].copy() # Added .copy()
vaccination_trend['Date'] = pd.to_datetime(vaccination_trend['Date'])

plt.figure(figsize=(14, 8))
for country in countries_to_plot:
    country_data = vaccination_trend[vaccination_trend['Country/Region'] == country]
    if not country_data.empty: # Ensure there's data to plot
        plt.plot(country_data['Date'], country_data['Vaccination Rate'], label=country)

plt.title('Vaccination Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Vaccination Rate (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(visualizations_folder, "vaccination_rate_over_time.png"))
plt.show()

print(f"\nAll visualizations have been attempted to be saved in the '{visualizations_folder}' folder.")