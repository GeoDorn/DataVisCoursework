import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def setup_directories(base_dir=None):
    """
    Set up the directory structure for data processing
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory path. If None, uses the current directory
    
    Returns:
    --------
    dict
        Dictionary containing paths to data directories
    """
    if base_dir is None:
        # Use a relative path structure if no base_dir is provided
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directory paths
    data_root = os.path.join(base_dir, 'data')
    raw_data = os.path.join(data_root, 'raw')
    cleaned_root = os.path.join(data_root, 'cleaned')
    processed_root = os.path.join(data_root, 'processed')
    
    # Create directories if they don't exist
    for directory in [cleaned_root, processed_root]:
        os.makedirs(directory, exist_ok=True)
    
    paths = {
        'data_root': data_root,
        'raw_data': raw_data,
        'cleaned_root': cleaned_root,
        'processed_root': processed_root,
        'jhu_data': os.path.join(raw_data, 'JHU'),
        'worldbank_data': os.path.join(raw_data, 'WorldBank')
    }
    
    return paths

def process_jhu_data(paths):
    """
    Process Johns Hopkins University COVID-19 data files
    
    Parameters:
    -----------
    paths : dict
        Dictionary of directory paths
    
    Returns:
    --------
    pandas.DataFrame or None
        Processed JHU COVID-19 data, or None if processing fails
    """
    jhu_path = paths['jhu_data']
    logger.info(f"Processing JHU COVID-19 data from {jhu_path}")
    
    try:
        # Check which JHU files are available
        cases_file = os.path.join(jhu_path, 'time_series_covid19_confirmed_global.csv')
        deaths_file = os.path.join(jhu_path, 'time_series_covid19_deaths_global.csv')
        recovered_file = os.path.join(jhu_path, 'time_series_covid19_recovered_global.csv')
        vaccine_doses_file = os.path.join(jhu_path, 'Vaccine', 'time_series_covid19_vaccine_doses_admin_global.csv')
        vaccine_global_file = os.path.join(jhu_path, 'Vaccine', 'time_series_covid19_vaccine_global.csv')
        
        available_files = {}
        for name, file_path in [
            ('cases', cases_file),
            ('deaths', deaths_file),
            ('recovered', recovered_file),
            ('vaccine_doses', vaccine_doses_file),
            ('vaccine_global', vaccine_global_file)
        ]:
            if os.path.exists(file_path):
                available_files[name] = file_path
                logger.info(f"Found {name} data: {file_path}")
            else:
                logger.warning(f"{name} data file not found: {file_path}")
        
        if not available_files:
            logger.error("No JHU data files found")
            return None
        
        # Process each file and convert from wide to long format
        datasets = {}
        for name, file_path in available_files.items():
            # Read the data
            df = pd.read_csv(file_path)
            
            # Check if the data is in time series format (dates as columns)
            date_cols = [col for col in df.columns if ('/' in col or '-' in col)]
            
            if date_cols:
                # Melt the data from wide to long format
                id_vars = [col for col in df.columns if col not in date_cols]
                
                melted_df = pd.melt(
                    df,
                    id_vars=id_vars,
                    value_vars=date_cols,
                    var_name='date',
                    value_name=f'{name}_count'
                )
                
                # Convert date to datetime format
                melted_df['date'] = pd.to_datetime(melted_df['date'])
                
                # Check if ISO codes are present, if not, create dummy iso codes
                if 'iso3' not in melted_df.columns and 'Country/Region' in melted_df.columns:
                    melted_df['iso_code'] = melted_df['Country/Region'].str.upper().str[:3]
                elif 'iso3' in melted_df.columns:
                    melted_df = melted_df.rename(columns={'iso3': 'iso_code'})
                
                # Standardize country column name
                if 'Country/Region' in melted_df.columns:
                    melted_df = melted_df.rename(columns={'Country/Region': 'location'})
                
                datasets[name] = melted_df
                logger.info(f"Processed {name} data: {melted_df.shape}")
        
        # Merge all datasets on common keys
        if datasets:
            # Start with the first dataset
            merged_df = datasets[list(datasets.keys())[0]]
            
            # Merge with the rest
            for name, df in list(datasets.items())[1:]:
                # Determine the common columns to merge on
                common_cols = [col for col in merged_df.columns if col in df.columns and col != f'{name}_count']
                
                if not common_cols:
                    logger.warning(f"No common columns found for merging {name} data")
                    continue
                
                # Merge
                merged_df = pd.merge(
                    merged_df,
                    df,
                    on=common_cols,
                    how='outer'
                )
            
            # Calculate additional metrics
            if 'cases_count' in merged_df.columns and 'deaths_count' in merged_df.columns:
                merged_df['case_fatality_rate'] = np.where(
                    merged_df['cases_count'] > 0,
                    merged_df['deaths_count'] / merged_df['cases_count'] * 100,
                    np.nan
                )
            
            # Calculate daily counts from cumulative
            for metric in ['cases', 'deaths', 'recovered']:
                if f'{metric}_count' in merged_df.columns:
                    # Calculate daily values
                    merged_df[f'new_{metric}'] = merged_df.groupby(['iso_code', 'location'])[f'{metric}_count'].diff().fillna(0)
                    
                    # Calculate 7-day moving averages
                    merged_df[f'new_{metric}_7day_avg'] = merged_df.groupby(['iso_code', 'location'])[f'new_{metric}'].transform(
                        lambda x: x.rolling(7, min_periods=1).mean()
                    )
            
            # Save the processed data
            processed_jhu_path = os.path.join(paths['cleaned_root'], 'cleaned_jhu_covid_data.csv')
            merged_df.to_csv(processed_jhu_path, index=False)
            logger.info(f"Processed JHU data saved to: {processed_jhu_path}")
            
            return merged_df
        
        else:
            logger.error("Failed to process JHU data files")
            return None
            
    except Exception as e:
        logger.error(f"Error processing JHU data: {str(e)}")
        return None

def process_economic_data(paths):
    """
    Process World Bank economic indicators data
    
    Parameters:
    -----------
    paths : dict
        Dictionary of directory paths
    
    Returns:
    --------
    pandas.DataFrame or None
        Processed economic data, or None if processing fails
    """
    worldbank_path = paths['worldbank_data']
    logger.info(f"Processing World Bank economic data from {worldbank_path}")
    
    try:
        # Check which World Bank files are available
        economic_file = os.path.join(worldbank_path, 'economicIndicatorsData.csv')
        metadata_file = os.path.join(worldbank_path, 'econmicIndicatorsMetadata.csv')
        
        if not os.path.exists(economic_file):
            logger.error(f"Economic data file not found: {economic_file}")
            return None
        
        # Load economic data
        economic_data = pd.read_csv(economic_file)
        logger.info(f"Loaded economic data: {economic_data.shape}")
        
        # Load metadata if available
        metadata = None
        if os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file)
            logger.info(f"Loaded economic metadata: {metadata.shape}")
        
        # Check if data is in wide format (years as columns) and reshape if needed
        year_cols = [str(year) for year in range(2020, 2024)]
        has_year_cols = any(year in economic_data.columns for year in year_cols)
        
        if has_year_cols and 'indicator_name' in economic_data.columns:
            logger.info("Converting economic data from wide to long format")
            
            # List of year columns that actually exist in the data
            existing_year_cols = [col for col in year_cols if col in economic_data.columns]
            
            # Melt the dataframe to convert from wide to long format
            economic_data = pd.melt(
                economic_data,
                id_vars=[col for col in economic_data.columns if col not in existing_year_cols],
                value_vars=existing_year_cols,
                var_name='year',
                value_name='value'
            )
            
            # Convert year to datetime (first day of the year)
            economic_data['date'] = pd.to_datetime(economic_data['year'], format='%Y')
        
        # If indicators are in separate rows, pivot them to columns
        if 'indicator_name' in economic_data.columns:
            logger.info("Pivoting economic indicators from rows to columns")
            
            # Create a mapping of indicator names to standardized column names
            indicator_mapping = {
                'GDP growth (annual %)': 'gdp_growth',
                'GDP per capita (constant 2015 US$)': 'gdp_per_capita',
                'Unemployment, total (% of total labor force)': 'unemployment_rate',
                'Inflation, consumer prices (annual %)': 'inflation_rate',
                'General government final consumption expenditure (% of GDP)': 'govt_spending',
                'Central government debt, total (% of GDP)': 'govt_debt',
                'Current account balance (% of GDP)': 'current_account',
                'Foreign direct investment, net inflows (% of GDP)': 'fdi_inflows',
                'Exports of goods and services (% of GDP)': 'exports',
                'Imports of goods and services (% of GDP)': 'imports'
                # Add more mappings for other indicators as needed
            }
            
            # Create a new column with standardized names where possible
            if 'indicator_code' not in economic_data.columns:
                economic_data['indicator_code'] = economic_data['indicator_name'].map(indicator_mapping)
                # For indicators not in our mapping, use a cleaned version of the original name
                economic_data['indicator_code'] = economic_data['indicator_code'].fillna(
                    economic_data['indicator_name'].str.lower().str.replace('[^a-z0-9]', '_', regex=True)
                )
            
            # Identify columns for the index in the pivot
            index_cols = []
            for col in ['iso_code', 'country_code', 'country', 'country_name', 'date']:
                if col in economic_data.columns:
                    index_cols.append(col)
            
            # Add region/income group if available
            for col in ['region', 'income_group']:
                if col in economic_data.columns:
                    index_cols.append(col)
            
            # Pivot each indicator to become a column
            economic_data = economic_data.pivot_table(
                index=index_cols,
                columns='indicator_code',
                values='value',
                aggfunc='first'  # Use first to avoid aggregation issues
            ).reset_index()
            
            # Flatten the multi-level column names that result from pivot_table
            economic_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in economic_data.columns]
        
        # Standardize country code column
        if 'iso_code' not in economic_data.columns:
            if 'country_code' in economic_data.columns:
                economic_data['iso_code'] = economic_data['country_code'].str.upper()
            elif 'iso3' in economic_data.columns:
                economic_data['iso_code'] = economic_data['iso3'].str.upper()
        
        # Standardize country name column
        if 'location' not in economic_data.columns:
            if 'country' in economic_data.columns:
                economic_data['location'] = economic_data['country']
            elif 'country_name' in economic_data.columns:
                economic_data['location'] = economic_data['country_name']
        
        # Filter to 2020-2023 time period if date column exists
        if 'date' in economic_data.columns:
            economic_data = economic_data[(economic_data['date'] >= '2020-01-01') & 
                                       (economic_data['date'] <= '2023-12-31')]
        
        # Save the processed economic data
        processed_econ_path = os.path.join(paths['cleaned_root'], 'cleaned_economic_data.csv')
        economic_data.to_csv(processed_econ_path, index=False)
        logger.info(f"Processed economic data saved to: {processed_econ_path}")
        
        return economic_data
    
    except Exception as e:
        logger.error(f"Error processing economic data: {str(e)}")
        return None

def merge_datasets(covid_data, economic_data, paths):
    """
    Merge COVID-19 and economic datasets
    
    Parameters:
    -----------
    covid_data : pandas.DataFrame
        Processed COVID-19 data
    economic_data : pandas.DataFrame
        Processed economic data
    paths : dict
        Dictionary of directory paths
    
    Returns:
    --------
    pandas.DataFrame or None
        Merged dataset, or None if merging fails
    """
    logger.info("Attempting to merge COVID-19 and economic datasets")
    
    try:
        if covid_data is None or economic_data is None:
            logger.error("Cannot merge datasets: One or both input datasets are missing")
            return None
        
        # Determine common frequency for merging
        # Check if economic data has a date column
        if 'date' in economic_data.columns:
            # Determine the frequency of economic data
            has_daily_data = False
            
            # Check if economic data is grouped by year
            yearly_pattern = economic_data['date'].dt.is_year_start.sum() == len(economic_data)
            
            if yearly_pattern:
                freq = 'Y'
                logger.info("Economic data appears to be annual. Aggregating COVID data to annual.")
            else:
                # Default to monthly if we can't determine
                freq = 'M'
                logger.info("Economic data frequency uncertain. Aggregating COVID data to monthly.")
            
            # Aggregate covid data to match economic data frequency
            if freq != 'D':
                # Extract the matching period for grouping
                if freq == 'Y':
                    covid_data['period'] = covid_data['date'].dt.year
                elif freq == 'Q':
                    covid_data['period'] = covid_data['date'].dt.to_period('Q').astype(str)
                elif freq == 'M':
                    covid_data['period'] = covid_data['date'].dt.to_period('M').astype(str)
                
                # Identify columns to aggregate
                cumulative_cols = [col for col in covid_data.columns if col.endswith('_count')]
                new_cols = [col for col in covid_data.columns if col.startswith('new_')]
                rate_cols = [col for col in covid_data.columns if col.endswith('_rate')]
                
                # Create aggregation dictionary
                agg_dict = {}
                
                # Cumulative counts: use max for the period
                for col in cumulative_cols:
                    agg_dict[col] = 'max'
                
                # New counts: sum for the period
                for col in new_cols:
                    agg_dict[col] = 'sum'
                
                # Rates: use mean for the period
                for col in rate_cols:
                    agg_dict[col] = 'mean'
                
                # Aggregate
                covid_data_agg = covid_data.groupby(['iso_code', 'location', 'period']).agg(agg_dict).reset_index()
                
                # Create a date column for the aggregated data
                if freq == 'Y':
                    covid_data_agg['date'] = pd.to_datetime(covid_data_agg['period'], format='%Y')
                else:
                    # For quarterly and monthly, take the end date of the period
                    covid_data_agg['date'] = pd.PeriodIndex(covid_data_agg['period']).to_timestamp('E')
                
                covid_data = covid_data_agg
        
        # Prepare for merging
        # Ensure we have consistent column names for merging
        merge_cols = []
        if 'iso_code' in covid_data.columns and 'iso_code' in economic_data.columns:
            merge_cols.append('iso_code')
        
        if 'date' in covid_data.columns and 'date' in economic_data.columns:
            merge_cols.append('date')
        
        if not merge_cols:
            logger.error("No common columns found for merging datasets")
            return None
        
        # Merge the datasets
        merged_data = pd.merge(
            covid_data,
            economic_data,
            on=merge_cols,
            how='outer',
            suffixes=('_covid', '_econ')
        )
        
        # Handle duplicated columns from the merge
        if 'location_covid' in merged_data.columns and 'location_econ' in merged_data.columns:
            # Use the more complete column
            loc_covid_missing = merged_data['location_covid'].isna().sum()
            loc_econ_missing = merged_data['location_econ'].isna().sum()
            
            if loc_covid_missing <= loc_econ_missing:
                merged_data['location'] = merged_data['location_covid'].fillna(merged_data['location_econ'])
            else:
                merged_data['location'] = merged_data['location_econ'].fillna(merged_data['location_covid'])
                
            # Remove the duplicated columns
            merged_data = merged_data.drop(['location_covid', 'location_econ'], axis=1)
        
        # Save the merged dataset
        merged_data_path = os.path.join(paths['processed_root'], 'covid_economic_merged_data.csv')
        merged_data.to_csv(merged_data_path, index=False)
        logger.info(f"Merged dataset saved to: {merged_data_path}")
        logger.info(f"Final merged data shape: {merged_data.shape}")
        
        return merged_data
    
    except Exception as e:
        logger.error(f"Error while merging datasets: {str(e)}")
        return None

def main():
    """Main function to run the data processing pipeline"""
    logger.info("Starting COVID-19 data processing pipeline")
    
    # Setup directories
    paths = setup_directories()
    
    # Process JHU COVID-19 data
    covid_data = process_jhu_data(paths)
    
    # Process World Bank economic data
    economic_data = process_economic_data(paths)
    
    # Merge the datasets
    merged_data = merge_datasets(covid_data, economic_data, paths)
    
    # Final summary
    if merged_data is not None:
        logger.info("Data processing completed successfully!")
        
        # Print summary statistics
        logger.info("\nSummary of processed data:")
        if covid_data is not None:
            logger.info(f"COVID-19 data: {covid_data.shape[0]} rows, {covid_data.shape[1]} columns")
            logger.info(f"  Date range: {covid_data['date'].min()} to {covid_data['date'].max()}")
            logger.info(f"  Countries: {covid_data['iso_code'].nunique()}")
        
        if economic_data is not None:
            logger.info(f"Economic data: {economic_data.shape[0]} rows, {economic_data.shape[1]} columns")
            if 'date' in economic_data.columns:
                logger.info(f"  Date range: {economic_data['date'].min()} to {economic_data['date'].max()}")
            logger.info(f"  Countries: {economic_data['iso_code'].nunique()}")
        
        logger.info(f"Merged data: {merged_data.shape[0]} rows, {merged_data.shape[1]} columns")
    else:
        logger.error("Data processing failed")

if __name__ == "__main__":
    main()