import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
os.makedirs(f"{base_dir}/outputs", exist_ok=True)

orders = pd.read_csv(f"{base_dir}/data/clean/orders.csv", parse_dates=["order_ts"])
completed = orders[orders["status"] == "completed"]         
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = completed.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"))
rfm["recency"] = (snapshot - rfm["last_order"]).dt.days      
rfm = rfm.drop(columns=["last_order"]).reset_index()
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print("customers:", len(rfm))
print("recency median/min/max:", rfm["recency"].median(), rfm["recency"].min(), rfm["recency"].max())
print("frequency mean/max:", round(rfm["frequency"].mean(), 2), rfm["frequency"].max())
print("monetary mean/max:", round(rfm["monetary"].mean(), 2), round(rfm["monetary"].max(), 2))

scaler = StandardScaler()
scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

rows = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
    rows.append({"K": k, "inertia": round(model.inertia_, 1),
                 "silhouette": round(silhouette_score(scaled, model.labels_), 3)})
print(pd.DataFrame(rows).to_string(index=False))

CHOSEN_K = 4
final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42).fit(scaled)
rfm["segment_id"] = final.labels_

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")).round(2)
print(profile)


pca = PCA(n_components=2).fit(scaled)
print("PCA ratios:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))


premium_id = profile["mean_monetary"].idxmax()        
loyal_id = profile["mean_frequency"].idxmax()         
lapsing_id = profile["mean_recency"].idxmax()         

names = {}
for seg_id in profile.index:
    if seg_id == premium_id:
        names[seg_id] = "Premium Big Spenders"
    elif seg_id == loyal_id:
        names[seg_id] = "Loyal Regulars"
    elif seg_id == lapsing_id:
        names[seg_id] = "Lapsing Low-Value"
    else:
        names[seg_id] = "Occasional Buyers"

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print(rfm.groupby(["segment_id", "segment_name", "is_premium"]).size())

rfm = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]]
rfm.to_csv(f"{base_dir}/outputs/customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv", rfm.shape)

print("Increasing the number of centroids reduces the distance between each centroid and hence the sum of squared distances decreases and hence inertia of K=8 is lower than K=8")
