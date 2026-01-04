import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
import joblib
from scipy.stats import randint as sp_randint

try:
    import xgboost as xgb
    XGB = True
except:
    XGB = False

try:
    import shap
    SHAP = True
except:
    SHAP = False

def download(url, dest):
    dest = Path(dest)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)

def load_data():
    os.makedirs("data", exist_ok=True)
    red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    red_path = "data/winequality-red.csv"
    white_path = "data/winequality-white.csv"
    download(red_url, red_path)
    download(white_url, white_path)
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    return pd.concat([red, white]).reset_index(drop=True)

data = load_data()
data["quality_label"] = (data["quality"] >= 7).astype(int)

numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
hist_cols = [c for c in numeric_cols if c not in ["quality", "quality_label"]]

os.makedirs("figures", exist_ok=True)
for col in hist_cols:
    plt.figure()
    data[col].hist(bins=30)
    plt.title(col)
    plt.savefig(f"figures/{col}.png")
    plt.close()

try:
    import seaborn as sns
    plt.figure(figsize=(12,10))
    sns.heatmap(data.corr(), annot=False)
    plt.savefig("figures/correlation_matrix.png")
    plt.close()
except:
    pass

X = data.drop(columns=["quality", "quality_label"])
X = pd.get_dummies(X)
y_reg = data["quality"]
y_clf = data["quality_label"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_r_s = scaler.fit_transform(X_train_r)
X_test_r_s  = scaler.transform(X_test_r)
X_train_c_s = scaler.fit_transform(X_train_c)
X_test_c_s  = scaler.transform(X_test_c)

lr = LinearRegression()
lr.fit(X_train_r_s, y_train_r)
lr_pred = lr.predict(X_test_r_s)

ridge = Ridge()
ridge.fit(X_train_r_s, y_train_r)
ridge_pred = ridge.predict(X_test_r_s)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_r, y_train_r)
rf_pred = rf.predict(X_test_r)

reg_results = pd.DataFrame([
    ["LinearRegression", np.sqrt(mean_squared_error(y_test_r, lr_pred)), r2_score(y_test_r, lr_pred)],
    ["Ridge", np.sqrt(mean_squared_error(y_test_r, ridge_pred)), r2_score(y_test_r, ridge_pred)],
    ["RandomForest", np.sqrt(mean_squared_error(y_test_r, rf_pred)), r2_score(y_test_r, rf_pred)]
], columns=["Model", "RMSE", "R2"])

if XGB:
    xg_reg = xgb.XGBRegressor(n_estimators=200)
    xg_reg.fit(X_train_r, y_train_r)
    xg_pred = xg_reg.predict(X_test_r)
    reg_results.loc[len(reg_results)] = ["XGBoost", np.sqrt(mean_squared_error(y_test_r, xg_pred)), r2_score(y_test_r, xg_pred)]

reg_results.to_csv("regression_results_summary.csv", index=False)

log = LogisticRegression(max_iter=1000)
log.fit(X_train_c_s, y_train_c)
log_pred = log.predict(X_test_c_s)

rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train_c, y_train_c)
rfc_pred = rfc.predict(X_test_c)

clf_results = pd.DataFrame([
    ["LogisticRegression", accuracy_score(y_test_c, log_pred)],
    ["RandomForest", accuracy_score(y_test_c, rfc_pred)]
], columns=["Model", "Accuracy"])

if XGB:
    xg_clf = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss")
    xg_clf.fit(X_train_c, y_train_c)
    xg_clf_pred = xg_clf.predict(X_test_c)
    clf_results.loc[len(clf_results)] = ["XGBoost", accuracy_score(y_test_c, xg_clf_pred)]

clf_results.to_csv("classification_results_summary.csv", index=False)

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.to_csv("feature_importances_randomforest.csv")

if SHAP:
    explainer = shap.TreeExplainer(rf)
    vals = explainer.shap_values(X_test_r)
    shap.summary_plot(vals, X_test_r, show=False)
    plt.savefig("figures/shap_summary.png")
    plt.close()

param_dist = {
    'n_estimators': sp_randint(50, 400),
    'max_depth': sp_randint(3, 20),
    'min_samples_split': sp_randint(2, 20),
    'min_samples_leaf': sp_randint(1, 10)
}

rs = RandomizedSearchCV(RandomForestRegressor(), param_dist, n_iter=20, cv=3, scoring="neg_root_mean_squared_error", random_state=42)
rs.fit(X_train_r, y_train_r)

best_rfr = rs.best_estimator_
best_pred = best_rfr.predict(X_test_r)

joblib.dump(best_rfr, "best_random_forest_regressor.joblib")
joblib.dump(rf, "random_forest_regressor.joblib")
joblib.dump(lr, "linear_regression.joblib")
joblib.dump(log, "logistic_regression.joblib")
joblib.dump(scaler, "standard_scaler.joblib")

print("Project executed successfully. Outputs saved in folder.")
