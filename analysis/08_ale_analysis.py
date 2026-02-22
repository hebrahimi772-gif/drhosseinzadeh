import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sklearn.inspection import PartialDependenceDisplay


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    imp_path = Path(cfg["outputs"]["tables"]) / "shap_importance.csv"
    if not imp_path.exists():
        raise FileNotFoundError("SHAP importance file not found")
    importance = pd.read_csv(imp_path)
    # convert feature names e.g. 'feat_0' to integer indices
    def parse_feat(f):
        try:
            return int(f.split("_")[1])
        except Exception:
            return f
    top_features = [parse_feat(f) for f in importance.feature.tolist()[:3]]

    raw = pd.read_csv(cfg["data"]["raw"])
    split = pd.read_csv(Path(cfg["outputs"]["model_artifacts"]) / "holdout_split.csv")
    train_ids = split.loc[split.split == "train", "id"]
    df = raw[raw.id.isin(train_ids)].copy()
    X = df.drop(columns=[cfg["target"], "id"])
    y = df[cfg["target"]]

    os.makedirs(cfg["outputs"]["figures"], exist_ok=True)
    for feat in top_features:
        fig, ax = plt.subplots()
        fm = joblib.load(Path(cfg["outputs"]["model_artifacts"]) / "final_model.pkl")
        if isinstance(fm, tuple):
            _, model = fm
        else:
            model = fm
        PartialDependenceDisplay.from_estimator(
            estimator=model,
            X=X,
            features=[feat],
            ax=ax,
        )
        fig.savefig(Path(cfg["outputs"]["figures"]) / f"ale_{feat}.png")
    print("ALE plots saved")


if __name__ == "__main__":
    import joblib
    main()
