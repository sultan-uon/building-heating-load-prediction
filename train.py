"""Compare four regressors and save the best Heating Load pipeline."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
SEED = 42
FEATURES = [
    "relative_compactness", "surface_area", "wall_area", "roof_area",
    "overall_height", "orientation", "glazing_area", "glazing_area_distribution",
]
CATEGORICAL = ["orientation", "glazing_area_distribution"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]


def load_data():
    data = pd.read_excel(ROOT / "data/ENB2012_data.xlsx")
    data = data[[f"X{i}" for i in range(1, 9)] + ["Y1"]].copy()
    data.columns = FEATURES + ["heating_load"]
    if data.isna().any().any() or data.duplicated().any():
        raise ValueError("Inspect missing or duplicate rows before training.")
    return data


def make_pipeline(name, estimator):
    # Encoders and scalers learn only from each fold's training rows.
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler() if name == "linear" else "passthrough", NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def train():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data.heating_load, test_size=0.2, random_state=SEED,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    estimators = {
        "linear": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=SEED),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1,
            objective="reg:squarederror", tree_method="hist", device="cpu",
            random_state=SEED, n_jobs=1),
    }
    pipelines = {name: make_pipeline(name, model) for name, model in estimators.items()}
    baseline = {}
    for name, pipeline in pipelines.items():
        scores = -cross_val_score(pipeline, X_train, y_train, cv=cv,
                                 scoring="neg_root_mean_squared_error", n_jobs=1)
        baseline[name] = {"cv_rmse": float(scores.mean()), "cv_std": float(scores.std())}
        print(name, baseline[name], flush=True)

    grids = {
        "random_forest": {"model__max_depth": [8, None], "model__min_samples_leaf": [1, 2, 4]},
        "xgboost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.1],
                    "model__n_estimators": [100, 300]},
    }
    tuned = {}
    candidates = {name: clone(pipeline) for name, pipeline in pipelines.items()}
    selection_scores = {name: row["cv_rmse"] for name, row in baseline.items()}
    for name, grid in grids.items():
        search = GridSearchCV(pipelines[name], grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        tuned[name] = {"cv_rmse": float(-search.best_score_), "parameters": search.best_params_}
        print(name, tuned[name], flush=True)
        if tuned[name]["cv_rmse"] <= selection_scores[name]:
            candidates[name] = search.best_estimator_
            selection_scores[name] = tuned[name]["cv_rmse"]

    # Dict order breaks exact ties. Test results never choose the winner.
    selected = min(selection_scores, key=selection_scores.get)
    model = candidates[selected].fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    selected_params = {key: value for key, value in model.get_params().items()
                       if key in grids.get(selected, {})}
    results = {"seed": SEED, "train_rows": len(X_train), "test_rows": len(X_test),
               "baseline": baseline, "tuned": tuned, "selected_model": selected,
               "selected_parameters": selected_params, "selected_cv_rmse": selection_scores[selected],
               "test_metrics": metrics}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/heating_model.joblib")
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    train()
