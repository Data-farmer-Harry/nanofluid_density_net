import os
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

USE_LOG_Y = False 


def load_data():
    train_npz = np.load(os.path.join("data", "train.npz"))["data"]
    test_npz  = np.load(os.path.join("data", "test.npz"))["data"]

    X_train = train_npz[:, :-1]
    y_train = train_npz[:, -1]

    X_test = test_npz[:, :-1]
    y_test = test_npz[:, -1]

    if USE_LOG_Y:
        y_train = np.log1p(y_train)
        y_test_for_eval = y_test  # calculation of the original scale MAE
    else:
        y_test_for_eval = y_test

    return X_train, y_train, X_test, y_test_for_eval


def main():
    X_train, y_train, X_test, y_test = load_data()

    # absoluteerror
    objective = "reg:absoluteerror"
    try:
        _ = xgb.XGBRegressor(objective=objective)
    except Exception:
        objective = "reg:pseudohubererror"

    base_model = xgb.XGBRegressor(
        objective=objective,
        eval_metric="mae",
        tree_method="hist",
        random_state=2022,
        n_jobs=-1,
    )

    # RandomizedSearchCV + 5-fold
    param_distributions = {
        "n_estimators": [300, 600, 900, 1200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        "min_child_weight": [1, 3, 5, 8],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0.0, 0.1, 0.2, 0.5],

        # Regular expressions 
        "reg_alpha": [0.0, 1e-3, 1e-2, 0.1],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "max_leaves": [0, 31, 63, 127],  
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=1,
        random_state=2022,
        n_jobs=-1,
    )

    print("Start XGBoost hyper-parameter search...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    if USE_LOG_Y:
        y_pred = np.expm1(y_pred)  # Restore to original scale

    mae = mean_absolute_error(y_test, y_pred)
    print(f"XGBoost Test MAE (best model): {mae:.4f}")

    out_dir = "train//XGBoost"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "xgb_model.json")
    best_model.save_model(model_path)
    print(f"XGBoost model saved to: {model_path}")

    # log1p
    meta = {"use_log_y": USE_LOG_Y, "objective": objective}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
