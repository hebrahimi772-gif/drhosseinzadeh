import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    raw = pd.read_csv(cfg["data"]["raw"])
    split = pd.read_csv(Path(cfg["outputs"]["model_artifacts"]) / "holdout_split.csv")
    train_ids = split.loc[split.split == "train", "id"]
    df = raw[raw.id.isin(train_ids)].copy()

    X = df.drop(columns=[cfg["target"], "id"])
    y = df[cfg["target"]]
    groups = df[cfg["group_var"]]

    # load preprocessing pipeline
    pipe = joblib.load(Path(cfg["outputs"]["model_artifacts"]) / "preprocessing_pipeline.pkl")
    from sklearn.base import clone as sklearn_clone

    outer_kf = GroupKFold(n_splits=cfg["cv"]["outer_folds"])
    results = []

    param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

    for fold, (train_index, test_index) in enumerate(outer_kf.split(X, y, groups=groups), start=1):
        X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_train = groups.iloc[train_index]

        # fit pipeline on training fold and transform both
        pipe_fold = sklearn_clone(pipe)
        X_train = pipe_fold.fit_transform(X_train_raw)
        X_test = pipe_fold.transform(X_test_raw)

        inner_kf = GroupKFold(n_splits=cfg["cv"]["inner_folds"])
        model = RandomForestRegressor(random_state=cfg.get("random_seed"))
        gs = GridSearchCV(model, param_grid, cv=inner_kf.split(X_train, y_train, groups_train), scoring=cfg["cv"]["scoring_metrics"], refit=cfg["cv"]["scoring_metrics"][0])
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        outer_r2 = r2_score(y_test, y_pred)
        outer_mse = mean_squared_error(y_test, y_pred)
        results.append({"fold": fold, "best_params": gs.best_params_, "r2": outer_r2, "mse": outer_mse})

    out_dir = Path(cfg["outputs"]["model_artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "nested_cv_results.csv", index=False)
    print(f"Nested CV results saved to {out_dir / 'nested_cv_results.csv'}")

    # train final model on full training data with pipeline
    X_full = pipe.fit_transform(X)
    final_model = RandomForestRegressor(random_state=cfg.get("random_seed"), **gs.best_params_)
    final_model.fit(X_full, y)
    joblib.dump((pipe, final_model), out_dir / "final_model.pkl")


if __name__ == "__main__":
    main()
