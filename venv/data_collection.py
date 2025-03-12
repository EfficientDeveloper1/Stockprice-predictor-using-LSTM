import yfinance as yf
import mysql.connector
import pandas as pd

#  Define stock tickers and date range
tickers = ["TSLA", "AAPL", "GOOGL", "NVDA", "AMZN"]
start_date = "2010-01-01"
end_date = "2024-12-31"

# Fetching and saving data
for ticker in tickers:
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        print(f"⚠️ No data found for {ticker}. Skipping...")
        continue

    df.reset_index(inplace=True)
    df["Ticker"] = ticker 
    
    filename = f"{ticker}_stock_data.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Data for {ticker} saved to {filename}")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata"
)
cursor = db.cursor()

# ✅ Function to create table
def create_table(cursor, table_name):
    query = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}` (
        `Date` DATE NOT NULL,
        `Ticker` VARCHAR(10) NOT NULL,
        `Open` FLOAT,
        `High` FLOAT,
        `Low` FLOAT,
        `Close` FLOAT,
        `Volume` INT,
        PRIMARY KEY (`Date`, `Ticker`)
    )
    """
    cursor.execute(query)
    print(f"✅ Table '{table_name}' ensured.")

# ✅ Function to insert CSV data into MySQL
def insert_csv_to_mysql(cursor, db, file_path, table_name):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date 
    
    inserted_rows = 0
    for _, row in df.iterrows():
        try:
            query = f"""
            INSERT IGNORE INTO `{table_name}` (`Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            values = (row["Date"], row["Ticker"], row["Open"], row["High"], row["Low"], row["Close"], row["Volume"])
            cursor.execute(query, values)
            inserted_rows += 1
        except Exception as e:
            print(f"⚠️ Skipped row due to error: {e}")

    db.commit()
    print(f"✅ Inserted {inserted_rows} rows from '{file_path}' into '{table_name}'.")

# ✅ Upload CSV files
create_table(cursor, "stock_prices")

tickers = ["TSLA", "AAPL", "GOOGL", "NVDA", "AMZN"]
for ticker in tickers:
    file_path = f"{ticker}_stock_data.csv"
    insert_csv_to_mysql(cursor, db, file_path, "stock_prices")

# ✅ Close MySQL connection
cursor.close()
db.close()

