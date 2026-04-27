import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

# Load data
data = np.load("ecg_splits.npz")

X_train = data["X_train"]
X_val   = data["X_val"]
X_test  = data["X_test"]

y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]

print("Loaded splits!")
print(X_train.shape, y_train.shape)


# Define search space
xgb_param_dist = {
    "n_estimators": [200, 300, 350],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.3],
    "reg_lambda": [1.0, 1.5, 2.0],
}

# Model
xgb_base = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    tree_method="hist",
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=1
)

# Search
xgb_search = RandomizedSearchCV(
    xgb_base,
    xgb_param_dist,
    n_iter=10,
    cv=3,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)


# Results
print(f"\nBest XGB params: {xgb_search.best_params_}")
print(f"Best CV F1: {xgb_search.best_score_:.4f}")


# Evaluate
xgb_tuned = xgb_search.best_estimator_

y_pred_xgb_tuned = xgb_tuned.predict(X_val)

print("\n" + "="*60)
print("          CLASSIFICATION REPORT: XGBOOST TUNED")
print("="*60)

print(classification_report(y_val, y_pred_xgb_tuned, digits=4))

print("="*60)