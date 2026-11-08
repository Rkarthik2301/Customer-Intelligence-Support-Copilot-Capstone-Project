import pandas as pd 
import numpy as np
import sqlite3
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
customers = pd.read_csv(f"{base_dir}/data/raw/customers.csv")
orders = pd.read_csv(f"{base_dir}/data/raw/orders.csv")
products = pd.read_csv(f"{base_dir}/data/raw/products.csv")
print("Loaded:", len(customers), "customers,", len(orders), "orders,", len(products), "products")

#Auditing data
print("TABLE CUSTOMERS")
print("\nShape of customers table:", customers.shape)
print("\nData types:\n",customers.dtypes)
print("\n Missing Values\n",customers.isnull().sum())
print("\n Missing Values in percentage\n",customers.isnull().mean() * 100)
print("\nNumber of blank age:",customers["age"].isna().sum())
print("\n Number of Invalid Age:",((pd.to_numeric(customers["age"],errors="coerce")<10)|(pd.to_numeric(customers["age"],errors="coerce")>100)).sum())
print("\n Number of distinct raw city spellings:",customers["city"].dropna().nunique())
print("\n Number of distinct non-blank raw gender spellings:", customers["gender"].dropna().nunique())
print("\n Number of blank gender:", customers["gender"].isna().sum())

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

#cleaning Customers.csv
customers["city"] = customers["city"].str.strip().str.title()

customers["gender"] = customers["gender"].fillna("").str.strip().str.lower()
gender_map = {"m": "male", "male": "male", "f": "female", "female": "female"}
customers["gender"] = customers["gender"].map(gender_map)    
customers["gender"] = customers["gender"].fillna("unknown")  

too_young = customers["age"] < 10
too_old = customers["age"] > 100
customers.loc[too_young | too_old, "age"] = np.nan

city_median_age = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_age)
customers["age"] = customers["age"].astype(int)

customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Cleaning Orders.csv
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

orders = orders.drop_duplicates()

orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset="order_id", keep="last")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
rebuilt = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(rebuilt)

orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
is_completed = orders["status"] == "completed"
completed_only = orders.where(is_completed)                    # non-completed rows become NaN
category_median = completed_only.groupby("category")["delivery_minutes"].transform("median")
filled = orders["delivery_minutes"].fillna(category_median)
orders["delivery_minutes"] = orders["delivery_minutes"].where(~is_completed, filled)

orders = orders.drop(columns=["category", "price_inr"])
orders = orders.sort_values("order_id").reset_index(drop=True)

#Saving files
customers.to_csv(f"{base_dir}/data/clean/customers.csv", index=False)
products.to_csv(f"{base_dir}/data/clean/products.csv", index=False)
orders.to_csv(f"{base_dir}/data/clean/orders.csv", index=False)

# SQL connect

customers_db = customers.copy()
orders_db = orders.copy()
customers_db["signup_date"] = customers_db["signup_date"].dt.strftime("%Y-%m-%d")
orders_db["order_ts"] = orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

connection = sqlite3.connect(f"{base_dir}/data/instamart.db")
customers_db.to_sql("customers", connection, if_exists="replace", index=False)
products.to_sql("products", connection, if_exists="replace", index=False)
orders_db.to_sql("orders", connection, if_exists="replace", index=False)
connection.close()
print("Done. Clean files and data/instamart.db written.")