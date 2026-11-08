import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
orders = pd.read_csv(f"{base_dir}/data/raw/orders.csv")
customers = pd.read_csv(f"{base_dir}/data/raw/customers.csv")
products = pd.read_csv(f"{base_dir}/data/raw/products.csv")

#Auditing data
print("TABLE CUSTOMERS")
print("\nShape of customers table:", customers.shape)
print("\nData types:\n",customers.dtypes)
print("\n Missing Values\n",customers.isnull().sum())
print("\n Missing Values in percentage\n",customers.isnull().mean() * 100)
print("\nNumber of blank age:",customers["age"].isna().sum())
print("\n Number of Invalid Age:",((pd.to_numeric(customers["age"],errors="coerce")<10)|(pd.to_numeric(customers["age"],errors="coerce")>100)).sum())
print("\n Number of distinct raw city spellings:",customers["city"].dropna().nunique())
print("\n Number of distinct non-blank raw gender spellings:",customers["gender"].isna().sum())

print("\nTABLE ORDERS")
print("\nShape of orders table:", orders.shape)
print("\nData types:\n",orders.dtypes)
print("\n Missing Values\n",orders.isnull().sum())
print("\n Missing Values in percentage\n",orders.isnull().mean() * 100)
print("Raw data rows:\n",orders.to_string)
print("\nNumber of Duplicates:",orders.duplicated().sum())
print("\nNumber of rows with 'Pending' status:",(orders["status"]=="pending").sum())
print("\nDistinct order ids :",orders["order_id"].nunique())
print("\nOrder entries that cannnot be converted to numbers:",pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum())
numeric_minutes = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Number of negative time:",(numeric_minutes < 0).sum())
print("Number of blank minutes:",orders["delivery_minutes"].isna().sum())


print("\nTABLE PRODUCTS")
print("\nShape of products table:", products.shape)
print("\nData types:\n",products.dtypes)
print("\n Missing Values\n",products.isnull().sum())
print("\n Missing Values in percentage\n",products.isnull().mean() * 100)

#Running the seven SQL commands
connection = sqlite3.connect(f"{base_dir}/data/instamart.db")
text = open(f"{base_dir}/sql/queries.sql").read()
pieces = text.split("-- Q")[1:] 
queries = []
for piece in pieces:
    sql = piece.split("\n", 1)[1].strip()        # remove the "1" left from "-- Q1"
    queries.append(sql)
    
results = []
for sql in queries:
    results.append(pd.read_sql_query(sql, connection))

q1, q2, q3, q4, q5, q6, q7 = results

print("\n=== SQL RESULTS")
print("Q1 customers:", len(q1))
print("Q2 rows:", len(q2), "| total INR:", round(q2["revenue"].sum(), 2))
print("Q2 largest cell:", q2.iloc[0].tolist())
print("Q3 products above average price:", len(q3))
print("Q4 customers with no completed order:", len(q4))
print("Q5 customers above average spend:", len(q5), "| average INR:", round(q5["avg_total"].iloc[0], 2))
print("Q6 top customer in each city:")
print(q6[q6["row_num"] == 1][["customer_id", "city", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK differs from ROW_NUMBER:", (q6["rnk"] != q6["row_num"]).sum())
print("Q7 daily rows:", len(q7), "| last cumulative INR:", round(q7["cumulative_revenue"].iloc[-1], 2))

#PANDAS CROSS-CHECK
data = pd.read_sql_query("""
    SELECT o.*, c.city, p.category
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed'""", connection)

pivot = pd.pivot_table(data, values="order_amount", index="city", columns="category",
                       aggfunc="sum", margins=True)
pivot_total = round(pivot.loc["All", "All"], 2)
sql_total = round(q2["revenue"].sum(), 2)
print("\n=== PIVOT grand total:", pivot_total, "| SQL total:", sql_total)
assert pivot_total == sql_total, "pandas and SQL totals do not match!"

per_city = data.groupby("city")["order_amount"].agg(total="sum", average="mean", orders="count")
print(per_city.round(2))

#Data Exploration
x = data["order_amount"]
q1_value = x.quantile(0.25)
q3_value = x.quantile(0.75)
IQR = q3_value - q1_value
lower_fence = max(0,q1_value - 1.5 * IQR) #since order amount cannot be less than 0
upper_fence = q3_value + 1.5 * IQR

print("\n order_amount statistics")
print(" mean:", round(x.mean(), 2))  
print(" median:", x.median())
print(" skewness:", round(x.skew(), 2))
print("Q1:", q1_value) 
print("Q3:", q3_value)
print("lower fence:", lower_fence)
print("upper fence:", upper_fence)
print("outliers above:", (x > upper_fence).sum(), "| below:", (x < lower_fence).sum())

correlations = data[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(correlations.round(3))

best_pair, best_value = None, 0
columns = list(correlations.columns)
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        value = correlations.iloc[i, j]
        if abs(value) > abs(best_value):
            best_pair, best_value = (columns[i], columns[j]), value


#Question and Answers
print("\n Q1: The data is about the details of each orders like order amount,quantity,which customer orederd (order id)")
print("\n Q2: The data looks a little bit right-skewed")
print("\n Q3: There are 897 outliers above the upper fence but none below the lower fence")
print("\n Q4: There is a strong negative correlation between order_amount and discount_pct and a positive correlation between order_amount and quantity")

print("largest correlation pair:", best_pair,",Largest correlation value:", round(best_value, 3))