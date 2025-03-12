import mysql.connector
import pandas as pd

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Marvelous@32",
    database="stockdata"
)
cursor = db.cursor()

# Fetching data from MySQL
query = "SELECT * FROM stock_prices" 
cursor.execute(query)
data = cursor.fetchall()

# get column names from my table
columns = [i[0] for i in cursor.description]

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Close DB connection
cursor.close()
db.close()

# Display the first few rows
print(df.head())

# Checking for any missing values
print("\nMissing Values:\n", df.isnull().sum())

# Checking data types
print("\nData Types:\n", df.dtypes)
