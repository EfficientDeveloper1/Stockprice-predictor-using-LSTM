import pandas as pd
import mysql.connector as connect

#Connect to MySQL
db = connect.connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata"
)
cursor = db.cursor()

#Fetches Data with Correct Date Range
query = "SELECT * FROM stock_prices WHERE Date >= '2010-01-01'"
df = pd.read_sql(query, db)

#Ensures Date is Datetime and Sorted
df["Date"] = pd.to_datetime(df["Date"])
df.sort_values(by=["Ticker", "Date"], inplace=True)

#My Features engineering

# Simple Moving Average (SMA)
df["SMA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=10).mean())
df["SMA_30"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=30).mean())

# Exponential Moving Average (EMA)
df["EMA_10"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=10, adjust=False).mean())
df["EMA_30"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=30, adjust=False).mean())

# Bollinger Bands
df["Rolling_Mean"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=20).mean())
df["Rolling_Std"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=20).std())
df["Upper_Band"] = df["Rolling_Mean"] + (df["Rolling_Std"] * 2)
df["Lower_Band"] = df["Rolling_Mean"] - (df["Rolling_Std"] * 2)

# Daily Returns
df["Daily_Return"] = df.groupby("Ticker")["Close"].pct_change()

# Relative Strength Index (RSI)
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI"] = df.groupby("Ticker")["Close"].transform(lambda x: compute_rsi(x))

# Average True Range (ATR) - Volatility
df["High_Low"] = df["High"] - df["Low"]
df["High_Close"] = (df["High"] - df["Close"].shift()).abs()
df["Low_Close"] = (df["Low"] - df["Close"].shift()).abs()
df["True_Range"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
df["ATR"] = df.groupby("Ticker")["True_Range"].transform(lambda x: x.rolling(window=14).mean())

# Drop Temporary Calculation Columns
df.drop(columns=["High_Low", "High_Close", "Low_Close", "True_Range"], inplace=True)

# Ensure stock_features Table Only Has 2010+ Data
cursor.execute("DELETE FROM stock_features WHERE Date < '2010-01-01';")
db.commit()

# Insert Features into MySQL
def insert_features_to_mysql(cursor, db, df, table_name):
    if df.empty:
        print("⚠️ No data to insert! DataFrame is empty.")
        return

    inserted_rows = 0
    print(f"📊 First few rows of DataFrame to insert:\n{df.head()}")  # Debugging step

    for _, row in df.iterrows():
        try:
            # ✅ Convert Date to string for MySQL
            date_value = row["Date"].strftime("%Y-%m-%d")

            query = f"""
            INSERT INTO {table_name} (`Date`, `Ticker`, `SMA_10`, `SMA_30`, `EMA_10`, `EMA_30`, 
                                      `Upper_Band`, `Lower_Band`, `Daily_Return`, `RSI`, `ATR`)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                SMA_10 = VALUES(SMA_10),
                SMA_30 = VALUES(SMA_30),
                EMA_10 = VALUES(EMA_10),
                EMA_30 = VALUES(EMA_30),
                Upper_Band = VALUES(Upper_Band),
                Lower_Band = VALUES(Lower_Band),
                Daily_Return = VALUES(Daily_Return),
                RSI = VALUES(RSI),
                ATR = VALUES(ATR);
            """

            values = (
                date_value, row["Ticker"], row["SMA_10"], row["SMA_30"], 
                row["EMA_10"], row["EMA_30"], row["Upper_Band"], row["Lower_Band"], 
                row["Daily_Return"], row["RSI"], row["ATR"]
            )

            cursor.execute(query, values)
            inserted_rows += 1
        except Exception as e:
            print(f"⚠️ Skipped row due to error: {e}")

    db.commit()
    print(f"✅ Inserted {inserted_rows} rows into '{table_name}'.")

#Calls the Function to Insert Data
insert_features_to_mysql(cursor, db, df, "stock_features")

#closing connection
cursor.close()
db.close()
