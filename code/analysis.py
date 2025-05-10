import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams.update({'font.size': 12})

def clean_data(df):
    """Clean and prepare the data for analysis."""
    # Convert string columns to appropriate types
    numeric_cols = ['Confirmed', 'Deaths', 'People_at_least_one_dose', 
                   'Population', 'Doses Administered', 'GDP growth (annual %)',
                   'Inflation, consumer prices (annual %)', 'Population, total',
                   'Unemployment, total (% of total labor force) (national estimate)']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle any 'W' or other data issues in the rows
    if df['Country/Region'].dtype == object:  # Check if it's a string column
        df = df[~df['Country/Region'].astype(str).str.contains('W')]
    
    # Add derived metrics
    if all(col in df.columns for col in ['Confirmed', 'Population']):
        # Avoid division by zero
        mask = (df['Population'] > 0) & df['Population'].notna() & df['Confirmed'].notna()
        df.loc[mask, 'Cases_per_100k'] = (df.loc[mask, 'Confirmed'] / df.loc[mask, 'Population']) * 100000
    
    if all(col in df.columns for col in ['Deaths', 'Confirmed']):
        # Avoid division by zero
        mask = (df['Confirmed'] > 0) & df['Confirmed'].notna() & df['Deaths'].notna()
        df.loc[mask, 'Fatality_Rate'] = (df.loc[mask, 'Deaths'] / df.loc[mask, 'Confirmed']) * 100
    
    if all(col in df.columns for col in ['People_at_least_one_dose', 'Population']):
        # Avoid division by zero
        mask = (df['Population'] > 0) & df['Population'].notna() & df['People_at_least_one_dose'].notna()
        df.loc[mask, 'Vaccination_Rate'] = (df.loc[mask, 'People_at_least_one_dose'] / df.loc[mask, 'Population']) * 100
    
    # Replace infinities and extreme values
    for col in df.select_dtypes(include=[np.number]).columns:
        # Replace inf and -inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Optional: Cap extreme values (using 3 standard deviations from mean)
        if df[col].notna().any():  # Only process if we have valid data
            mean = df[col].mean()
            std = df[col].std()
            if not pd.isna(mean) and not pd.isna(std) and std > 0:
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                # Cap extreme values
                df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def correlation_analysis(df):
    """Perform correlation analysis on the dataset."""
    # Select numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
    
    plt.title('Correlation Matrix of COVID-19 and Economic Indicators', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed correlation analysis with p-values
    correlation_results = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            # Skip if either column has all NaN values
            if df[col1].notna().sum() > 1 and df[col2].notna().sum() > 1:
                valid_data = df[[col1, col2]].dropna()
                if len(valid_data) > 1:  # Need at least 2 points for correlation
                    corr, p_value = pearsonr(valid_data[col1], valid_data[col2])
                    correlation_results.append({
                        'Variable 1': col1,
                        'Variable 2': col2,
                        'Correlation': corr,
                        'P-value': p_value,
                        'Significance': '**' if p_value < 0.01 else ('*' if p_value < 0.05 else '')
                    })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('Correlation', key=abs, ascending=False)
    
    return corr_matrix, correlation_df

def linear_regression_analysis(df):
    """Perform linear regression analysis on key variables."""
    regression_results = []
    
    # Define key variable pairs to analyze
    variable_pairs = [
        ('Vaccination_Rate', 'Cases_per_100k'),
        ('GDP growth (annual %)', 'Cases_per_100k'),
        ('Vaccination_Rate', 'Fatality_Rate'),
        ('GDP growth (annual %)', 'Unemployment, total (% of total labor force) (national estimate)')
    ]
    
    # Create subplots figure
    fig = plt.figure(figsize=(16, 12))
    
    for i, (x_var, y_var) in enumerate(variable_pairs):
        if x_var in df.columns and y_var in df.columns:
            try:
                # Filter out rows with missing values, infinities, or extreme values
                valid_data = df[[x_var, y_var]].dropna()
                
                # Extra safety check for infinities or very large values
                valid_data = valid_data[~np.isinf(valid_data[x_var])]
                valid_data = valid_data[~np.isinf(valid_data[y_var])]
                
                # Skip if there's not enough data
                if len(valid_data) <= 2:
                    print(f"Skipping regression for {x_var} vs {y_var}: insufficient data points")
                    continue
                
                # Log message about data size
                print(f"Linear regression for {x_var} vs {y_var}: {len(valid_data)} valid data points")
                
                # Remove outliers (optional)
                # For x variable
                Q1_x = valid_data[x_var].quantile(0.25)
                Q3_x = valid_data[x_var].quantile(0.75)
                IQR_x = Q3_x - Q1_x
                valid_data = valid_data[(valid_data[x_var] >= Q1_x - 1.5 * IQR_x) & 
                                       (valid_data[x_var] <= Q3_x + 1.5 * IQR_x)]
                
                # For y variable
                Q1_y = valid_data[y_var].quantile(0.25)
                Q3_y = valid_data[y_var].quantile(0.75)
                IQR_y = Q3_y - Q1_y
                valid_data = valid_data[(valid_data[y_var] >= Q1_y - 1.5 * IQR_y) & 
                                       (valid_data[y_var] <= Q3_y + 1.5 * IQR_y)]
                
                # Skip if there's still not enough data after outlier removal
                if len(valid_data) <= 2:
                    print(f"Skipping regression for {x_var} vs {y_var}: insufficient data points after outlier removal")
                    continue
                
                X = valid_data[x_var].values.reshape(-1, 1)
                y = valid_data[y_var].values
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate R-squared
                y_pred = model.predict(X)
                r_squared = model.score(X, y)
                
                # Store results
                regression_results.append({
                    'Independent Variable': x_var,
                    'Dependent Variable': y_var,
                    'Coefficient': model.coef_[0],
                    'Intercept': model.intercept_,
                    'R-squared': r_squared,
                    'Sample Size': len(valid_data)
                })
                
                # Plot regression line
                ax = fig.add_subplot(2, 2, i+1)
                ax.scatter(X, y, alpha=0.7)
                ax.plot(X, y_pred, color='red', linewidth=2)
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.set_title(f'Linear Regression: {x_var} vs {y_var}\nR² = {r_squared:.4f}')
                
            except Exception as e:
                print(f"Error in regression analysis for {x_var} vs {y_var}: {e}")
    
    plt.tight_layout()
    plt.savefig('linear_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    regression_df = pd.DataFrame(regression_results) if regression_results else pd.DataFrame()
    
    return regression_df

def clustering_analysis(df):
    """Perform K-means clustering on the dataset."""
    try:
        # Select features for clustering
        cluster_features = ['Cases_per_100k', 'Fatality_Rate', 'Vaccination_Rate', 
                            'GDP growth (annual %)', 'Inflation, consumer prices (annual %)']
        
        # Check if features exist in dataframe
        available_features = [col for col in cluster_features if col in df.columns]
        print(f"Available features for clustering: {available_features}")
        
        if len(available_features) < 2:
            print("Not enough features available for clustering")
            return None
        
        # Filter rows with complete data for clustering
        cluster_data = df[available_features].dropna()
        
        # Check for infinities and replace with NaN
        for col in cluster_data.columns:
            mask = np.isinf(cluster_data[col])
            if mask.any():
                print(f"Removing {mask.sum()} infinity values from {col}")
                cluster_data.loc[mask, col] = np.nan
        
        # Drop rows with NaN after handling infinities
        original_len = len(cluster_data)
        cluster_data = cluster_data.dropna()
        if len(cluster_data) < original_len:
            print(f"Dropped {original_len - len(cluster_data)} rows with NaN values")
        
        if len(cluster_data) < 3:
            print("Not enough complete data points for clustering")
            return None
        
        print(f"Number of data points for clustering: {len(cluster_data)}")
        
        # Handle outliers for each feature using IQR method
        for feature in available_features:
            Q1 = cluster_data[feature].quantile(0.25)
            Q3 = cluster_data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((cluster_data[feature] < lower_bound) | (cluster_data[feature] > upper_bound)).sum()
            if outliers > 0:
                print(f"Feature '{feature}' has {outliers} outliers")
                
                # Cap outliers instead of removing them
                cluster_data[feature] = cluster_data[feature].clip(lower_bound, upper_bound)
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Check for any remaining issues in the scaled data
        if np.isnan(scaled_data).any() or np.isinf(scaled_data).any():
            print("Warning: Scaled data still contains NaN or infinity values")
            # Replace any remaining problematic values
            scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        max_clusters = min(8, len(cluster_data) - 2)  # Limit max clusters
        
        if max_clusters < 2:
            print("Not enough data points to determine optimal number of clusters")
            return None
        
        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                score = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(score)
                print(f"Silhouette score for k={k}: {score:.4f}")
            except Exception as e:
                print(f"Error calculating silhouette score for k={k}: {e}")
                silhouette_scores.append(-1)  # Use a negative score to indicate failure
        
        if not silhouette_scores or max(silhouette_scores) < 0:
            print("Failed to calculate valid silhouette scores")
            return None
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method for Optimal K')
        plt.grid(True)
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Choose optimal k (maximum silhouette score)
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Apply K-means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df_clustered = cluster_data.copy()
        df_clustered['Cluster'] = kmeans.fit_predict(scaled_data)
        
        # Visualize clusters using PCA if we have more than 2 features
        if len(available_features) > 2:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(scaled_data)
                
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                    c=df_clustered['Cluster'], cmap='viridis', 
                                    alpha=0.8, s=100, edgecolors='w')
                
                # Add country labels if available
                if 'Country/Region' in df.columns:
                    # Get country values safely
                    try:
                        countries = df.loc[cluster_data.index, 'Country/Region'].values
                        # Limit the number of annotations to prevent overcrowding
                        max_annotations = min(50, len(pca_result))
                        indices = np.linspace(0, len(pca_result)-1, max_annotations, dtype=int)
                        
                        for i in indices:
                            plt.annotate(countries[i], (pca_result[i, 0], pca_result[i, 1]), 
                                        fontsize=8, alpha=0.7, ha='center', va='bottom')
                    except Exception as e:
                        print(f"Error adding country labels: {e}")
                
                plt.colorbar(scatter, label='Cluster')
                plt.title(f'K-means Clustering (k={optimal_k}) - PCA Visualization', fontsize=16)
                plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error creating PCA visualization: {e}")
        
        # Calculate cluster profiles
        cluster_profiles = df_clustered.groupby('Cluster').mean()
        
        return df_clustered, cluster_profiles, optimal_k
    
    except Exception as e:
        print(f"Error in clustering analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the entire analysis."""
    print("Loading and cleaning data...")
    try:
        # Try to load the CSV file
        df = pd.read_csv('data/merged/combined_dataset.csv')
        print("Successfully loaded data from CSV file.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        print("Using example data instead.")
        # Fallback to example data if CSV loading fails
        example_data = """Country/Region,Date,Confirmed,Deaths,People_at_least_one_dose,Population,Doses Administered,GDP growth (annual %),"Inflation, consumer prices (annual %)","Population, total","Unemployment, total (% of total labor force) (national estimate)"
United Kingdom,2021-12-07,10620303.0,174736.0,51138245.0,67886004.0,237848147.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-08,10671356.0,174870.0,51161757.0,67886004.0,238734902.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-09,10721839.0,175002.0,51183457.0,67886004.0,239667758.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-10,10780442.0,175126.0,51207496.0,67886004.0,240722442.0,0.085759509048565,0.025183710961421298,67026292.0,0.0486
W United Kingdom,2021-12-11,10833032.0,175235.0,51229559.0,67886004.0,241697260.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-12,10881385.0,175372.0,51255850.0,67886004.0,242928779.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-13,10935598.0,175486.0,51279167.0,67886004.0,243830867.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865
United Kingdom,2021-12-14,10995158.0,175638.0,51298838.0,67886004.0,244956753.0,0.085759509048565,0.025183710961421298,67026292.0,0.04865"""
        # Create a DataFrame from the example data
        from io import StringIO
        df = pd.read_csv(StringIO(example_data))
    
    df = clean_data(df)
    
    print("\nBasic dataset information:")
    print(f"Shape: {df.shape}")
    print(f"Countries: {df['Country/Region'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\nPerforming correlation analysis...")
    corr_matrix, correlation_df = correlation_analysis(df)
    print("\nTop 5 strongest correlations:")
    print(correlation_df.head())
    
    print("\nPerforming linear regression analysis...")
    regression_df = linear_regression_analysis(df)
    print("\nRegression results:")
    print(regression_df)
    
    print("\nPerforming clustering analysis...")
    clustering_results = clustering_analysis(df)
    
    if clustering_results:
        df_clustered, cluster_profiles, optimal_k = clustering_results
        print(f"\nOptimal number of clusters: {optimal_k}")
        print("\nCluster profiles:")
        print(cluster_profiles)
        
        if 'Country/Region' in df.columns:
            print("\nCountries by cluster:")
            for cluster in range(optimal_k):
                countries = df.loc[df_clustered[df_clustered['Cluster'] == cluster].index, 'Country/Region'].unique()
                print(f"Cluster {cluster}: {', '.join(countries)}")
    
    print("\nAnalysis complete! Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()