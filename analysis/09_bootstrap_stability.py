import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor


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

    # apply preprocessing pipeline to features
    pipe = joblib.load(Path(cfg["outputs"]["model_artifacts"]) / "preprocessing_pipeline.pkl")
    # fit pipeline on full training features
    X_trans = pipe.fit_transform(X)
    # use generic feature names based on transformed dimension
    n_feats = X_trans.shape[1]
    features = [f"feat_{i}" for i in range(n_feats)]

    n_boot = 50
    imps = []
    rng = np.random.RandomState(cfg.get("random_seed"))
    unique_groups = groups.unique()
    for i in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = groups.isin(sampled)
        model = RandomForestRegressor(random_state=cfg.get("random_seed"))
        model.fit(X_trans[mask], y[mask])
        imps.append(model.feature_importances_)

    imps = np.vstack(imps)
    # regenerate feature names based on imps width to guarantee match
    features = [f"feat_{i}" for i in range(imps.shape[1])]
    summary = pd.DataFrame({"feature": features, "mean_importance": imps.mean(axis=0), "std_importance": imps.std(axis=0)})
    out_dir = Path(cfg["outputs"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "bootstrap_stability.csv", index=False)
    print("Bootstrap stability table written")


if __name__ == "__main__":
    main()
