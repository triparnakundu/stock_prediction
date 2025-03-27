import yfinance as yf

# Define the ETFs
etfs = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'SPY']

# Define start and end dates
start_date = "2000-01-01"
end_date = "2024-12-31"

# Loop through each ETF and download data
for etf in etfs:
    print(f"Downloading data for {etf}...")
    # Download the data
    data = yf.download(etf, start=start_date, end=end_date)
    
    # Reset the index to make Date a column
    data.reset_index(inplace=True)
    
    # Format the Date column to only include the date part
    data['Date'] = data['Date'].dt.date
    
    # Save the cleaned data to CSV
    filename = f"{etf}.csv"
    data.to_csv(filename, index=False)  # Ensure the index is not saved
    print(f"Cleaned data for {etf} saved to {filename}")

print("All data has been downloaded and saved.")
