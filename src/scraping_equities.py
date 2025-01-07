# Import necessary packages
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# List of tickers
stocks = ['QQQ', 'ASML', 'SAP', 'NVO', 'BESI.AS', 'ASM.AS', 'ASML.AS', 'MC.PA']

# Define the time range
start_date = datetime(2000, 1, 1)
end_date = datetime(2024, 11, 22)

# Create an empty DataFrame to store results
all_data = pd.DataFrame()

# Loop through each stock and collect its data
for ticker in stocks:
    print(f"Downloading data for {ticker}...")
    # Download daily data
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')['Close']
    # Rename the column to the ticker name
    data = data.rename(ticker)
    # Add to the main DataFrame
    if all_data.empty:
        all_data = data
    else:
        all_data = pd.merge(all_data, data, left_index=True, right_index=True, how='outer')

# Drop rows with all NaN values (if no data for a given day for all tickers)
all_data.dropna(how='all', inplace=True)

# Save the data to a CSV for future use
output_file = "../equities.csv"
all_data.to_csv(output_file)
print(f"Data saved to {output_file}")

# Display the first few rows of the consolidated DataFrame
print(all_data.head())

differences = all_data.dropna().pct_change()

differences = differences.dropna()