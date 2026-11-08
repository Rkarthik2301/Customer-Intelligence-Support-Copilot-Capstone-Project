import sqlite3
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
connection = sqlite3.connect(f"{base_dir}/data/instamart.db")

def one_number(sql):
    # run a SQL query that returns a single number and give that number back
    return connection.execute(sql).fetchone()[0]

checks = []
checks.append(("12309 order rows", one_number("SELECT COUNT(*) FROM orders") == 12309))
checks.append(("12309 distinct order_id", one_number("SELECT COUNT(DISTINCT order_id) FROM orders") == 12309))
checks.append(("11110 completed", one_number("SELECT COUNT(*) FROM orders WHERE status='completed'") == 11110))
checks.append(("1199 cancelled", one_number("SELECT COUNT(*) FROM orders WHERE status='cancelled'") == 1199))
checks.append(("no pending rows", one_number("SELECT COUNT(*) FROM orders WHERE status='pending'") == 0))
checks.append(("0 missing order_amount", one_number("SELECT COUNT(*) FROM orders WHERE order_amount IS NULL") == 0))
checks.append(("0 missing delivery minutes (completed)",
               one_number("SELECT COUNT(*) FROM orders WHERE status='completed' AND delivery_minutes IS NULL") == 0))
checks.append(("1199 missing delivery (cancelled)",
               one_number("SELECT COUNT(*) FROM orders WHERE status='cancelled' AND delivery_minutes IS NULL") == 1199))
revenue = one_number("SELECT SUM(order_amount) FROM orders WHERE status='completed'")
checks.append(("revenue is 2998668.80", round(revenue, 2) == 2998668.80))
checks.append(("6 cities", one_number("SELECT COUNT(DISTINCT city) FROM customers") == 6))
checks.append(("1200 customers", one_number("SELECT COUNT(*) FROM customers") == 1200))
checks.append(("0 missing age", one_number("SELECT COUNT(*) FROM customers WHERE age IS NULL") == 0))
checks.append(("ages between 18 and 60",
               one_number("SELECT MIN(age) >= 18 AND MAX(age) <= 60 FROM customers") == 1))

failed = []
passed_t = []
for name, passed in checks:
    if not passed:
        failed.append(name)
        
if len(failed) > 0:
    print("FAILED CHECKS:", failed)
    raise SystemExit(1)          

for name,passed in checks :
    if passed:
        passed_t.append(name)
        
if __name__ == "__main__" :
    print("Passed rubrics:",passed_t)