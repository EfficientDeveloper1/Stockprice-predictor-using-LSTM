import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt 


# ✅ Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata"
)
cursor = db.cursor()

# Fetch data from MySQL
query = "SELECT * FROM stock_prices"
df = pd.read_sql(query, db)


# Convert 'Date' to datetime and sort chronologically
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
df = df.sort_values(by=["Date", "Ticker"])

df = df[df["Date"] >= "2010-01-01"]

#check the last 10 dates
print("\n🔍 Checking Last 10 Dates in the Dataset:")
print(df[["Date", "Ticker"]].drop_duplicates().sort_values("Date").tail(10))

# Summary Statistics
summary_stats = df.describe()
print("\n📊 Summary Statistics:\n", df.describe())

#checking for possible ticker duplicates
print("\n Checking Unique Tickers:")

df["Ticker"] = df["Ticker"].str.strip()#removes white spaces
print (df["Ticker"].value_counts())

#drop duplicates based on date and ticker
df = df.drop_duplicates(subset= ["Date", "Ticker"]).reset_index(drop=True)

#ensures only 5 unique tickers exist
unique_tickers = df["Ticker"].unique()

if len(unique_tickers) > 5:
  print(f"warning: More than 5 tickers detected ({len(unique_tickers)})). Filtering first 5.")
  df = df[df["Ticker"].isin(unique_tickers[:5])]

# plot the closing prices of top 5 unique tickers
plt.figure(figsize=(12, 6))
for ticker, group in df.groupby("Ticker"):
  plt.plot(group["Date"], group["Close"], label=ticker)

#the clsing prices are plotted with correct alignment
plt.figure(figsize=(12, 6))

for ticker, group in df.groupby("Ticker"):
    plt.plot(group["Date"], group["Close"], label=ticker)  #ensures dates are properly sorted

plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.title("Closing Prices of Top 5 Unique Tickers")
plt.legend(title="Ticker", loc="upper left")

plt.show()

# ✅ Close DB connection
cursor.close()
db.close()
