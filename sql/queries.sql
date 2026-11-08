-- Q1
SELECT customer_id, SUM(order_amount) AS total FROM orders
WHERE status='completed' AND order_amount>500
GROUP BY customer_id HAVING SUM(order_amount)>8000;
-- Q2
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c ON o.customer_id=c.customer_id
JOIN products p ON o.product_id=p.product_id
WHERE o.status='completed' GROUP BY c.city, p.category ORDER BY revenue DESC;
-- Q3
SELECT * FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products);
-- Q4
SELECT c.customer_id FROM customers c
LEFT JOIN orders o ON o.customer_id=c.customer_id AND o.status='completed'
WHERE o.order_id IS NULL;
-- Q5
WITH spend AS (SELECT customer_id, SUM(order_amount) AS total FROM orders WHERE status='completed' GROUP BY customer_id),
avg_spend AS (SELECT AVG(total) AS avg_total FROM spend)
SELECT s.customer_id, s.total, a.avg_total FROM spend s, avg_spend a WHERE s.total>a.avg_total;
-- Q6
WITH spend AS (SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
FROM orders o JOIN customers c ON o.customer_id=c.customer_id WHERE o.status='completed' GROUP BY o.customer_id, c.city)
SELECT customer_id, city, total_spend,
RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend;
-- Q7
WITH daily AS (SELECT SUBSTR(order_ts,1,10) AS day, SUM(order_amount) AS revenue FROM orders WHERE status='completed' GROUP BY day)
SELECT day, revenue, SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue FROM daily;