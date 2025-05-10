import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Custom color palettes
covid_palette = sns.color_palette("RdYlBu_r", 10)
econ_palette = sns.color_palette("viridis", 10)
correlation_cmap = LinearSegmentedColormap.from_list("correlation_cmap", ["#d13b40", "#f9f9f9", "#2874A6"])

# Load data
data_filepath = 'data/merged/combined_dataset.csv'  # Update with actual file path
df = pd.read_csv(data_filepath)

# Data preprocessing
def preprocess_data(df):
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Fill missing values for economic indicators with forward fill
    economic_cols = ['GDP growth (annual %)', 'Inflation, consumer prices (annual %)', 
                     'Unemployment, total (% of total labor force) (national estimate)']
    
    for col in economic_cols:
        df[col] = df.groupby('Country/Region')[col].ffill()
    
    # Calculate per capita metrics
    df['Infections_per_100k'] = (df['Confirmed'] / df['Population']) * 100000
    df['Deaths_per_100k'] = (df['Deaths'] / df['Population']) * 100000
    df['Vaccination_rate'] = df['People_at_least_one_dose'] / df['Population']
    
    # Extract temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')
    df['Quarter'] = df['Date'].dt.to_period('Q')
    
    return df

# Process the data
df = preprocess_data(df)

# VISUALIZATION 1: HEATMAP of infection rates by country
def create_infection_heatmap():
    # Get latest data for each country
    latest_data = df.sort_values('Date').groupby('Country/Region').last().reset_index()
    
    # Select top countries by infection rate
    top_countries = latest_data.sort_values('Infections_per_100k', ascending=False).head(15)
    
    plt.figure(figsize=(10, 12))
    
    # Create dataframe for heatmap
    heatmap_data = pd.DataFrame({
        'Country': top_countries['Country/Region'],
        'Infections': top_countries['Infections_per_100k']
    }).set_index('Country')
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.1f', cbar_kws={'label': 'Infections per 100,000 people'})
    
    plt.title('COVID-19 Infection Rates by Country (per 100,000 population)', fontsize=16)
    plt.tight_layout()
    plt.savefig('infection_heatmap.png', dpi=300)
    plt.close()

# VISUALIZATION 2: LINE CHART comparing GDP growth trends
def create_gdp_comparison():
    # Select major economies for comparison
    major_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'China', 'Japan', 'Brazil', 'India']
    filtered_df = df[df['Country/Region'].isin(major_countries)]
    
    # Group by country and quarter for GDP
    quarterly_data = filtered_df.groupby(['Country/Region', 'Quarter'])['GDP growth (annual %)'].mean().reset_index()
    quarterly_data['Quarter'] = quarterly_data['Quarter'].astype(str)
    
    plt.figure(figsize=(14, 8))
    
    # Plot GDP trends
    for country, data in quarterly_data.groupby('Country/Region'):
        plt.plot(data['Quarter'], data['GDP growth (annual %)'], marker='o', linewidth=2, label=country)
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.title('GDP Growth Trends During COVID-19 Pandemic', fontsize=16)
    plt.xlabel('Quarter', fontsize=14)
    plt.ylabel('GDP Growth (annual %)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Annotate pandemic phases
    plt.axvspan('2020-Q1', '2020-Q2', alpha=0.2, color='red', label='Initial outbreak')
    plt.axvspan('2020-Q3', '2021-Q1', alpha=0.2, color='orange', label='Vaccine development')
    plt.axvspan('2021-Q2', '2022-Q2', alpha=0.2, color='green', label='Recovery phase')
    
    plt.tight_layout()
    plt.savefig('gdp_comparison.png', dpi=300)
    plt.close()

# VISUALIZATION 3: BAR CHART comparing vaccination rates and GDP growth
def create_vax_gdp_barchart():
    # Get latest data for selected countries
    latest_data = df.sort_values('Date').groupby('Country/Region').last().reset_index()
    
    # Filter for countries with complete data
    complete_data = latest_data.dropna(subset=['Vaccination_rate', 'GDP growth (annual %)'])
    
    # Select top 10 countries by vaccination rate
    top10_countries = complete_data.sort_values('Vaccination_rate', ascending=False).head(10)
    
    # Set up the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()
    
    # Plot vaccination rates as bars
    bars = ax1.barh(top10_countries['Country/Region'], top10_countries['Vaccination_rate'] * 100, 
                   color='steelblue', alpha=0.7, label='Vaccination Rate (%)')
    
    # Plot GDP growth as markers
    ax2.scatter(top10_countries['GDP growth (annual %)'], top10_countries['Country/Region'], 
               color='red', s=100, label='GDP Growth (%)')
    
    # Add labels and legend
    ax1.set_title('Vaccination Rates and GDP Growth in Top 10 Most Vaccinated Countries', fontsize=16)
    ax1.set_xlabel('Percentage', fontsize=14)
    ax1.set_xlim(0, 100)
    ax1.set_ylabel('Country', fontsize=14)
    
    # Add a vertical line at 0% GDP growth
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('GDP Growth (%)', fontsize=14)
    
    # Create combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('vax_gdp_comparison.png', dpi=300)
    plt.close()

# VISUALIZATION 4: INTEGRATED DASHBOARD
def create_dashboard():
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid for the dashboard
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Top-left: Infection trends
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Select major countries for infection trends
    major_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Italy']
    filtered_df = df[df['Country/Region'].isin(major_countries)]
    
    # Plot infection trends for each country
    for country, data in filtered_df.groupby('Country/Region'):
        # Group by month to reduce noise
        monthly_data = data.groupby('YearMonth').agg({'Infections_per_100k': 'max'}).reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'] + '-01')
        ax1.plot(monthly_data['Date'], monthly_data['Infections_per_100k'], label=country)
    
    ax1.set_title('COVID-19 Infection Rates Over Time', fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Infections per 100,000 people', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Top-right: Vaccination progress
    ax2 = fig.add_subplot(gs[0, 1])
    
    vax_df = df.dropna(subset=['Vaccination_rate'])
    latest_vax = vax_df.sort_values('Date').groupby('Country/Region').last().reset_index()
    top_vax_countries = latest_vax.sort_values('Vaccination_rate', ascending=False).head(10)
    
    # Create horizontal bar chart for vaccination rates
    bars = ax2.barh(top_vax_countries['Country/Region'], top_vax_countries['Vaccination_rate'] * 100, 
                  color=sns.color_palette("viridis", 10))
    
    ax2.set_title('Top 10 Countries by Vaccination Rate', fontsize=16)
    ax2.set_xlabel('Vaccination Rate (%)', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels to the bars
    for i, v in enumerate(top_vax_countries['Vaccination_rate']):
        ax2.text(v * 100 + 1, i, f"{v*100:.1f}%", va='center')
    
    # 3. Bottom-left: GDP growth vs Vaccination scatter plot
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Filter data for the scatter plot
    scatter_data = latest_vax.dropna(subset=['Vaccination_rate', 'GDP growth (annual %)'])
    scatter_data = scatter_data[scatter_data['Population'] > 10000000]  # Focus on larger countries
    
    # Create scatter plot
    scatter = ax3.scatter(scatter_data['Vaccination_rate'] * 100, 
                        scatter_data['GDP growth (annual %)'],
                        s=scatter_data['Population'] / 5000000,  # Size based on population
                        alpha=0.7,
                        c=scatter_data['Deaths_per_100k'],  # Color based on death rate
                        cmap='RdYlBu_r')
    
    # Add reference line at 0% GDP growth
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add country labels for selected points
    for idx, row in scatter_data.iterrows():
        if row['Country/Region'] in ['United States', 'United Kingdom', 'Germany', 'China', 'Japan', 'Brazil', 'India']:
            ax3.annotate(row['Country/Region'], 
                       (row['Vaccination_rate'] * 100, row['GDP growth (annual %)']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add color bar for death rates
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Deaths per 100,000 population', fontsize=12)
    
    # Add regression line
    x = scatter_data['Vaccination_rate'] * 100
    y = scatter_data['GDP growth (annual %)']
    
    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax3.plot(x, intercept + slope*x, 'r-', alpha=0.7)
    
    # Add regression statistics
    stat_text = f"R² = {r_value**2:.2f}\np-value = {p_value:.3f}\ny = {slope:.2f}x + {intercept:.2f}"
    ax3.text(0.05, 0.95, stat_text, transform=ax3.transAxes, 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    ax3.set_title('Vaccination Rate vs. GDP Growth', fontsize=16)
    ax3.set_xlabel('Vaccination Rate (%)', fontsize=12)
    ax3.set_ylabel('GDP Growth (annual %)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Bottom-right: Early vs Late adopters comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Define early adopters (countries that reached 20% vaccination by a certain date)
    vax_milestone = {}
    for country, group in df.dropna(subset=['Vaccination_rate']).groupby('Country/Region'):
        # Find the first date when vaccination rate exceeded 20%
        milestone_date = group[group['Vaccination_rate'] >= 0.2]['Date'].min()
        if pd.notna(milestone_date):
            vax_milestone[country] = milestone_date
    
    # Convert to DataFrame
    milestone_df = pd.DataFrame.from_dict(vax_milestone, orient='index', columns=['Milestone_Date'])
    milestone_df.reset_index(inplace=True)
    milestone_df.rename(columns={'index': 'Country/Region'}, inplace=True)
    
    # Find median date
    median_date = milestone_df['Milestone_Date'].median()
    
    # Categorize countries as early or late adopters
    milestone_df['Early_Adopter'] = milestone_df['Milestone_Date'] <= median_date
    
    # Get the latest data for each country
    latest_data = df.sort_values('Date').groupby('Country/Region').last().reset_index()
    
    # Merge with milestone data
    comparison_df = pd.merge(latest_data, milestone_df[['Country/Region', 'Early_Adopter']], 
                         on='Country/Region', how='inner')
    
    # Prepare data for grouped bar chart
    group_data = comparison_df.groupby('Early_Adopter').agg({
        'GDP growth (annual %)': 'mean',
        'Deaths_per_100k': 'mean',
        'Infections_per_100k': 'mean'
    }).reset_index()
    
    # Set up bar positions
    bar_width = 0.25
    r1 = np.arange(len(group_data))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create grouped bar chart
    bars1 = ax4.bar(r1, group_data['GDP growth (annual %)'], width=bar_width, label='GDP Growth (%)', color='green')
    bars2 = ax4.bar(r2, group_data['Deaths_per_100k']/50, width=bar_width, label='Deaths per 100k ÷ 50', color='red')
    bars3 = ax4.bar(r3, group_data['Infections_per_100k']/1000, width=bar_width, 
                  label='Infections per 100k ÷ 1000', color='orange')
    
    # Add labels
    ax4.set_title('Early vs. Late Vaccination Adoption Outcomes', fontsize=16)
    ax4.set_xticks([r + bar_width for r in range(len(group_data))])
    ax4.set_xticklabels(['Late Adopters', 'Early Adopters'])
    ax4.set_ylabel('Value', fontsize=12)
    ax4.legend(fontsize=10)
    
    # Add value labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.figtext(0.5, 0.01, 
              "Note: Death and infection rates have been scaled down for visualization purposes.", 
              ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('integrated_dashboard.png', dpi=300)
    plt.close()

# Execute all visualizations
create_infection_heatmap()
create_gdp_comparison()
create_vax_gdp_barchart()
create_dashboard()

# Optional: Additional visualizations for specific audiences

# For policymakers - Simplified high-level visual
def create_policymaker_visual():
    # Get latest data
    latest_data = df.sort_values('Date').groupby('Country/Region').last().reset_index()
    
    # Create vaccination quartiles
    latest_data['Vaccination_Quartile'] = pd.qcut(latest_data['Vaccination_rate'].fillna(0), 
                                               q=4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
    
    # Calculate average metrics by quartile
    quartile_data = latest_data.groupby('Vaccination_Quartile').agg({
        'GDP growth (annual %)': 'mean',
        'Deaths_per_100k': 'mean',
        'Unemployment, total (% of total labor force) (national estimate)': 'mean'
    }).reset_index()
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot GDP growth by vaccination quartile
    ax = sns.barplot(x='Vaccination_Quartile', y='GDP growth (annual %)', 
                   data=quartile_data, palette='RdYlGn')
    
    # Add data labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}%", 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha = 'center', va = 'bottom',
                  fontsize=12)
    
    plt.title('Economic Recovery by Vaccination Rate Quartile', fontsize=16)
    plt.xlabel('Vaccination Rate Quartile', fontsize=14)
    plt.ylabel('Average GDP Growth (annual %)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('policymaker_visual.png', dpi=300)
    plt.close()

create_policymaker_visual()