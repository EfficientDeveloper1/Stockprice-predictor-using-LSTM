import mysql.connector
import pandas as pd

# ✅ Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata"
)
cursor = db.cursor()

#  Fetch data from MySQL
query = "SELECT * FROM stock_prices"
cursor.execute(query)
data = cursor.fetchall()

#Get column names
columns = [i[0] for i in cursor.description]

#Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

#1. Convert 'Date' to datetime
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')  # Convert & handle errors

#2. If there is any duplicate row, it removes it
df.drop_duplicates(subset=["Ticker", "Date"], keep="first", inplace=True)

# 3. Remove invalid prices (negative values)
df = df[(df["Open"] >= 0) & (df["High"] >= 0) & (df["Low"] >= 0) & (df["Close"] >= 0)]

#  4. Ensure sorted order (by Ticker & Date)
df.sort_values(by=["Ticker", "Date"], inplace=True)

#  Display results
print(df.head())

#  Verify data types after cleaning
print("\nUpdated Data Types:\n", df.dtypes)

#saving the cleaned data back to the database

cursor.execute("DELETE FROM stock_prices")  # Clear existing table

for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO stock_prices (id, Ticker, Date, Open, High, Low, Close, Volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (row["id"], row["Ticker"], row["Date"].strftime("%Y-%m-%d"),
          row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]))

db.commit()


#  Close DB connection
cursor.close()
db.close()
