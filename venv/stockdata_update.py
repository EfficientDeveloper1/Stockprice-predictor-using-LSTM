import yfinance as yf
import pandas as pd
import mysql.connector as connector
import datetime
import os

#  MySQL Connection
db = connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata",
    allow_local_infile=True
)
cursor = db.cursor()

#  Ensure MySQL allows local file loading
cursor.execute("SET GLOBAL local_infile = 1")

#  Create Main Table (if not exists)
def create_main_table():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date DATE NOT NULL,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INT,
            Ticker VARCHAR(10),
            PRIMARY KEY (Date, Ticker)
        )
    """)
    db.commit()

#  it Creates Temporary Table (if it does not exist)
def create_temp_table():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices_temp (
            Date DATE NOT NULL,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INT,
            Ticker VARCHAR(10),
            PRIMARY KEY (Date, Ticker)
        )
    """)
    db.commit()

#  Fetches the Last Available Date
def get_last_date(cursor, ticker):
    query = "SELECT MAX(Date) FROM stock_prices WHERE Ticker = %s"
    cursor.execute(query, (ticker,))
    last_date = cursor.fetchone()[0]

    print(f"🔍 Last available date for {ticker} in database: {last_date}") 

    return last_date

#  Fetches Stock Data and Saves to CSV
def fetch_stock_data_to_csv(ticker):

    
    last_date = get_last_date(cursor, ticker)

    fixed_end_date = "2025-03-05"

    if last_date:
        start_date = (last_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        print(f"⚠ No previous data found for {ticker}. Fetching all data...")
        start_date = "2010-01-01"

    #end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    #print(f"🔄 Fetching new data for {ticker} from {start_date} to {end_date}...")
    end_date = fixed_end_date
    print(f" No previous data found for {ticker}. Fetching all data...")

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)

    if df.empty:
        print(f"⚠ No new data found for {ticker}. Skipping...")

    else:
        print(f" Sucessfully fetched {len(df)} new rows for {ticker}.")
        #start_date = (datetime.datetime.today() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
        #df = stock.history(start=start_date, end=end_date)

       #if df.empty:
           # print(f"��� No data found for {ticker} after trying earlier dates. Skipping...")
            #return None

    df.reset_index(inplace=True)
    df["Ticker"] = ticker

    #Converts Date to MySQL-compatible format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    #Saves Data to CSV
    file_path = f"{ticker}_stock_data.csv"
    df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]].to_csv(file_path, index=False)
    
    print(f"✅ CSV File Saved: {file_path}")
    return file_path

#Loads CSV Data into Temporary Table
def load_csv_to_temp_table(file_path):
    print(f"📥 Loading data from {file_path} into temporary MySQL table...")

    query = f"""
    LOAD DATA LOCAL INFILE '{file_path}'
    INTO TABLE stock_prices_temp
    FIELDS TERMINATED BY ',' 
    ENCLOSED BY '"' 
    LINES TERMINATED BY '\n'
    IGNORE 1 ROWS
    (Date, Open, High, Low, Close, Volume, Ticker);
    """

    cursor.execute(query)
    db.commit()

    print(f"✅ Data successfully loaded into temporary table.")

#Checks Data in Temporary Table Before Merging
def check_temp_table():
    cursor.execute("SELECT COUNT(*) FROM stock_prices_temp")
    count = cursor.fetchone()[0]
    print(f"📊 Records in stock_prices_temp before merging: {count}")

    if count > 0:
        cursor.execute("SELECT *FROM stock_prices_temp LIMIT 5")
        temp_data = cursor.fetchall()
        print(f"�� Sample Data from stock_prices_temp: {temp_data}")


    return count

#Merges Data from Temp Table to Main Table
def merge_temp_table():
    print("🔄 Merging data from temp table into main table...")

    #Checks if temp table has data before starting the merging
    if check_temp_table() == 0:
        print("⚠ No new data to merge. Skipping merge process.")
        return

    #Debugging: Print Temp Table Data Before Merge
    cursor.execute("SELECT * FROM stock_prices_temp LIMIT 5")
    temp_data = cursor.fetchall()
    print(f"📋 Sample Data from stock_prices_temp before merge: {temp_data}")

    query = """
    INSERT INTO stock_prices (Date, Open, High, Low, Close, Volume, Ticker)
    SELECT Date, Open, High, Low, Close, Volume, Ticker FROM stock_prices_temp
    ON DUPLICATE KEY UPDATE 
        Open = VALUES(Open),
        High = VALUES(High),
        Low = VALUES(Low),
        Close = VALUES(Close),
        Volume = VALUES(Volume);
    """

    try:
        cursor.execute(query)
        db.commit()
        print("✅ Data successfully merged into main table.")
    except Exception as e:
        print(f"❌ ERROR during merging: {e}")

    #Debugging: Check if data moved to stock_prices
    cursor.execute("SELECT COUNT(*) FROM stock_prices")
    count_after_merge = cursor.fetchone()[0]
    print(f"📊 Records in stock_prices after merging: {count_after_merge}")

#Cleanup: Drop Temporary Table
def drop_temp_table():
    cursor.execute("DROP TABLE IF EXISTS stock_prices_temp")
    db.commit()

#this process Stocks
tickers = ["TSLA", "AAPL", "GOOGL", "NVDA", "AMZN"]

#Create tables before inserting data
create_main_table()
create_temp_table()

for ticker in tickers:
    file_path = fetch_stock_data_to_csv(ticker)
    if file_path:
        load_csv_to_temp_table(file_path)
        os.remove(file_path)  #Deletes the CSV file after upload

#  Merges temp data into main table
merge_temp_table()

#  Cleanup: Remove temp table
drop_temp_table()

#  Close database connection
cursor.close()
db.close()
