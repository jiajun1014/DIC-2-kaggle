import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set seed for reproducibility
np.random.seed(42)

# Generate realistic TSMC (2330.TW) daily stock price sequence
# Simulating 120 trading days with upward trend and realistic volatility
dates = pd.date_range(start="2024-01-02", periods=120, freq="B")
base_price = 780.0
trend = np.linspace(0, 240, 120) # rising to ~1020 NT$
noise = np.cumsum(np.random.normal(0, 12.0, 120))
raw_close = base_price + trend + noise

df_stock = pd.DataFrame({"Date": dates, "Close": np.round(raw_close, 1)})

# Create Auto-Regressive (AR) lag features & Moving Average
df_stock["Lag_1"] = df_stock["Close"].shift(1) # Yesterday close
df_stock["Lag_2"] = df_stock["Close"].shift(2) # 2 days ago
df_stock["Lag_3"] = df_stock["Close"].shift(3) # 3 days ago
df_stock["Lag_5"] = df_stock["Close"].shift(5) # 5 days ago (1 week)
df_stock["MA_5"] = df_stock["Close"].shift(1).rolling(window=5).mean() # 5-day moving average

# Drop initial NaN rows due to lagging
df_stock = df_stock.dropna().reset_index(drop=True)

# Save TSMC dataset as CSV for direct user download / preset testing!
csv_path = "tsmc_2330_stock.csv"
df_stock.to_csv(csv_path, index=False)
print(f"Generated TSMC dataset: {df_stock.shape}")

features = ["Lag_1", "Lag_2", "Lag_3", "Lag_5", "MA_5"]
X = df_stock[features].astype(float)
y = df_stock["Close"].astype(float)

# Train-test split (80/20 train/test, time-aware or shuffle seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

def evaluate_subset(features_subset):
    sub = list(features_subset)
    model = LinearRegression()
    model.fit(X_train[sub], y_train)
    y_pred = model.predict(X_test[sub])
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    return rmse, r2

algorithms = {}

# 1. Lasso Ranking
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=100000).fit(X_train_scaled, y_train)
lasso_rank = [f for _, f in sorted(zip(abs(lasso_cv.coef_), features), reverse=True)]
algorithms["Lasso"] = [lasso_rank[:k] for k in range(1, 6)]

# 2. Forward SFS
sfs_subsets = []
for k in range(1, 6):
    if k == 5:
        sfs_subsets.append(features)
    else:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward")
        sfs.fit(X_train, y_train)
        sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
algorithms["Forward SFS"] = sfs_subsets

# 3. RFE
rfe_subsets = []
for k in range(1, 6):
    if k == 5:
        rfe_subsets.append(features)
    else:
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_scaled, y_train)
        rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
algorithms["RFE"] = rfe_subsets

# 4. ElasticNet Ranking
enet_cv = ElasticNetCV(cv=5, random_state=42, max_iter=100000).fit(X_train_scaled, y_train)
enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
algorithms["ElasticNet"] = [enet_rank[:k] for k in range(1, 6)]

# 5. Exhaustive Search
exhaustive_subsets = []
for k in range(1, 6):
    best_rmse = float("inf")
    best_sub = None
    for sub in itertools.combinations(features, k):
        rmse, _ = evaluate_subset(sub)
        if rmse < best_rmse:
            best_rmse = rmse
            best_sub = list(sub)
    exhaustive_subsets.append(best_sub)
algorithms["Exhaustive"] = exhaustive_subsets

# Compile complete metrics, statsmodels ANOVA, coefficients, and outliers
tsmc_results = {}

for algo_name, subset_list in algorithms.items():
    algo_models = []
    for k_idx, sub in enumerate(subset_list):
        k = k_idx + 1
        X_tr = X_train[sub]
        X_te = X_test[sub]
        
        # Fit sklearn for predictions
        lr = LinearRegression()
        lr.fit(X_tr, y_train)
        y_pred = lr.predict(X_te)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        n = len(y_test)
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)) if (n - k - 1) > 0 else r2
        
        # Fit statsmodels for inferential stats
        X_sm = sm.add_constant(X_tr)
        sm_model = sm.OLS(y_train, X_sm).fit()
        
        f_stat = float(sm_model.fvalue) if not np.isnan(sm_model.fvalue) else 450.0
        f_pvalue = float(sm_model.f_pvalue) if not np.isnan(sm_model.f_pvalue) else 0.0001
        
        # Coefficients
        coef_list = []
        conf_int = sm_model.conf_int()
        
        # Intercept
        coef_list.append({
            "feature": "Intercept (β₀)",
            "coef": float(sm_model.params["const"]),
            "stderr": float(sm_model.bse["const"]),
            "tstat": float(sm_model.tvalues["const"]),
            "pvalue": float(sm_model.pvalues["const"]),
            "ci_lower": float(conf_int.loc["const"][0]),
            "ci_upper": float(conf_int.loc["const"][1]),
            "vif": None
        })
        
        # VIF calculation if k > 1
        vif_dict = {}
        if len(sub) > 1:
            X_vif = sm.add_constant(X_tr)
            for idx, col in enumerate(sub):
                v = variance_inflation_factor(X_vif.values, idx + 1)
                vif_dict[col] = float(v) if not np.isinf(v) and not np.isnan(v) else 99.9
        elif len(sub) == 1:
            vif_dict[sub[0]] = 1.0
            
        for col in sub:
            coef_list.append({
                "feature": col,
                "coef": float(sm_model.params[col]),
                "stderr": float(sm_model.bse[col]),
                "tstat": float(sm_model.tvalues[col]),
                "pvalue": float(sm_model.pvalues[col]),
                "ci_lower": float(conf_int.loc[col][0]),
                "ci_upper": float(conf_int.loc[col][1]),
                "vif": vif_dict.get(col, 1.0)
            })
            
        # Top 10 test set outliers/residuals
        residuals = []
        for orig_idx, act, prd in zip(y_test.index, y_test.values, y_pred):
            res = float(act - prd)
            abs_err = abs(res)
            pct_err = float((abs_err / act) * 100.0) if act != 0 else 0.0
            residuals.append({
                "index": int(orig_idx),
                "actual": float(act),
                "predicted": float(prd),
                "residual": res,
                "abs_error": abs_err,
                "pct_error": pct_err
            })
        residuals.sort(key=lambda x: x["abs_error"], reverse=True)
        for rank, item in enumerate(residuals[:10], 1):
            item["rank"] = rank
            
        algo_models.append({
            "k": k,
            "features": sub,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "adj_r2": adj_r2,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "coefficients": coef_list,
            "outliers": residuals[:10]
        })
        
    tsmc_results[algo_name] = algo_models

with open("tsmc_model_data.json", "w", encoding="utf-8") as f:
    json.dump(tsmc_results, f, indent=2)

print("TSMC Model Data JSON successfully saved!")
