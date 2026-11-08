from pathlib import Path
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


base_dir = Path(__file__).resolve().parent.parent 
orders = pd.read_csv(f"{base_dir}/data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv(f"{base_dir}/data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(f"{base_dir}/data/clean/products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

before = orders[orders["order_ts"] < CUTOFF]
completed_before = before[before["status"] == "completed"]

c_ids = completed_before["customer_id"].unique() #Customers with atleast one order before cutoff

september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END) & (orders["status"] == "completed")]
ordered_in_september = set(september["customer_id"])
label = pd.DataFrame({"customer_id": c_ids})
label["lapsed"] = (~label["customer_id"].isin(ordered_in_september)).astype(int)

#features
feat = completed_before.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
    fav_category=("category", lambda s: s.value_counts().idxmax()))   # most frequent category
feat["recency"] = (CUTOFF - feat["last_order"]).dt.days
feat = feat.drop(columns=["last_order"]).reset_index()

#Cancellation rate 
final_status = before[before["status"].isin([ "completed","cancelled"])]
cancel_rate = final_status.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
cancel_rate = cancel_rate.rename("cancel_rate").reset_index()

data = label.merge(feat, on="customer_id").merge(cancel_rate, on="customer_id")
data = data.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
data["tenure_days"] = (CUTOFF - data["signup_date"]).dt.days

print("cohort size:", len(data))
print("lapsed:", data["lapsed"].sum(), f"({data['lapsed'].mean() * 100:.2f}%)")

numeric_cols = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
nominal_cols = ["city", "gender", "fav_category"]
X = data[numeric_cols + nominal_cols]
y = data["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train rows:", len(X_train), "| test rows:", len(X_test))

# baseline: always predict "not lapsed" (0)
baseline_pred = np.zeros(len(y_test), dtype=int)
print("baseline accuracy:", round(accuracy_score(y_test, baseline_pred) * 100, 1), "%",
      "| lapsed-class F1:", f1_score(y_test, baseline_pred, zero_division=0))

#Preprocess
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_cols)])

X_train_ready = preprocess.fit_transform(X_train)      
X_test_ready = preprocess.transform(X_test) 

scaler = preprocess.named_transformers_["num"]
freq_position = numeric_cols.index("frequency")
print("scaler mean of frequency (train only):", round(scaler.mean_[freq_position], 3))
print("full-cohort mean of frequency:", round(data["frequency"].mean(), 3))
print("encoded feature count:", X_train_ready.shape[1])

# no feature column may contain a missing value
print("missing values in raw features:", int(X.isna().sum().sum()))
print("NaN in encoded train/test:", int(np.isnan(X_train_ready).sum()), int(np.isnan(X_test_ready).sum()))

#Creating the models
def lapsed_scores(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    return {"precision": round(precision_score(y_eval, pred, zero_division=0), 3),
            "recall": round(recall_score(y_eval, pred, zero_division=0), 3),
            "f1": round(f1_score(y_eval, pred, zero_division=0), 3)}
    
log_pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
log_search = GridSearchCV(log_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
log_search.fit(X_train, y_train)
print("\nLOGISTIC best:", log_search.best_params_, "cv F1:", round(log_search.best_score_, 3))

rf_pipe = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_grid = {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}
rf_search = GridSearchCV(rf_pipe, rf_grid, cv=5, scoring="f1")
rf_search.fit(X_train, y_train)
print("FOREST best:", rf_search.best_params_, "cv F1:", round(rf_search.best_score_, 3))

log_model, rf_model = log_search.best_estimator_, rf_search.best_estimator_
for name, model in [("logistic", log_model), ("forest", rf_model)]:
    print(f"\n{name} test confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print(f"{name} lapsed-class scores:", lapsed_scores(model, X_test, y_test))
    
SELECTED = "forest" if lapsed_scores(rf_model, X_test, y_test)["f1"] >= lapsed_scores(log_model, X_test, y_test)["f1"] else "logistic"
selected_model = rf_model if SELECTED == "forest" else log_model
print("selected:", SELECTED)

#Over sampling the data fitting the model
train_df = X_train.copy()
train_df["lapsed"] = y_train
lapsed_rows = train_df[train_df["lapsed"] == 1]
kept_rows = train_df[train_df["lapsed"] == 0]
extra_needed = len(kept_rows) - len(lapsed_rows)
extra = lapsed_rows.sample(n=extra_needed, replace=True, random_state=42)   # duplicates, drawn with replacement
balanced = pd.concat([train_df, extra])                                      # keep ALL original rows + duplicates
print("\nbalanced train class counts:", balanced["lapsed"].value_counts().to_dict())

balanced_model = clone(log_model)                      # same best C, unfitted copy
balanced_model.fit(balanced[numeric_cols + nominal_cols], balanced["lapsed"])
print("logistic (original train):", lapsed_scores(log_model, X_test, y_test))
print("logistic (balanced train):", lapsed_scores(balanced_model, X_test, y_test))

#Interpretation
names = rf_model.named_steps["prep"].get_feature_names_out()
importances = pd.Series(rf_model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 forest features:\n", importances.head(5).round(3))
coefs = pd.Series(log_model.named_steps["clf"].coef_[0], index=names)
print("\nLogistic coefficients (sign):\n", coefs[["num__recency", "num__frequency", "num__monetary"]].round(3))

os.makedirs(f"{base_dir}/outputs", exist_ok=True)
model_path = f"{base_dir}/outputs/lapse_model.joblib"
joblib.dump(selected_model, model_path)
reloaded = joblib.load(model_path)
same = (reloaded.predict(X_test) == selected_model.predict(X_test)).all()
print("\nreloaded model reproduces test predictions exactly:", bool(same))
assert same

segments = pd.read_csv(f"{base_dir}/outputs/customer_segments.csv")
premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])
test_table = pd.DataFrame({"customer_id": data.loc[X_test.index, "customer_id"].values,
                           "lapse_probability": selected_model.predict_proba(X_test)[:, 1].round(4),
                           "actually_lapsed": y_test.values})
retention = test_table[test_table["customer_id"].isin(premium_ids)]
retention = retention.sort_values("lapse_probability", ascending=False).head(10)
retention.to_csv(f"{base_dir}/outputs/retention_list.csv", index=False)
print("retention list rows:", len(retention), "| truly lapsed:", int(retention["actually_lapsed"].sum()))