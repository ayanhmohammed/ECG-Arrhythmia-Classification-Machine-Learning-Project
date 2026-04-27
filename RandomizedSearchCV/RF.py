import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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
rf_param_dist = {
    'n_estimators': [250, 300, 350],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [6, 8, 10],
    'max_features': ['log2'],
}

# Model
rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)

# Search
rf_search = RandomizedSearchCV(
    rf_base,
    rf_param_dist,
    n_iter=8,        # smaller now
    cv=3,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)   #  fixed


# Results
print(f"\nBest RF params: {rf_search.best_params_}")
print(f"Best CV F1: {rf_search.best_score_:.4f}")


# Evaluate
from sklearn.metrics import classification_report

# 1. Get the best model from your search
rf_tuned = rf_search.best_estimator_

# 2. Run predictions on your validation set
y_pred_rf_tuned = rf_tuned.predict(X_val)

# 3. Print the report
print("\n" + "="*60)
print("             CLASSIFICATION REPORT: RF TUNED")
print("="*60)

# Use target_names if you have the labels (e.g., target_names=le.classes_)
print(classification_report(y_val, y_pred_rf_tuned, digits=4))

print("="*60)