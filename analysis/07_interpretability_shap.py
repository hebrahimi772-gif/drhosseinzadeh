import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import yaml


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
    feature_names = X.columns.tolist()

    pipe_model = joblib.load(Path(cfg["outputs"]["model_artifacts"]) / "final_model.pkl")
    # support tuple (pipe, model) or bare model
    if isinstance(pipe_model, tuple):
        pipe, model = pipe_model
        X_trans = pipe.transform(X)
    else:
        model = pipe_model
        X_trans = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # feature importance
    # use transformed feature count for names
    n_feats = shap_values.shape[1] if hasattr(shap_values, 'shape') else len(shap_values[0])
    feat_names = [f"feat_{i}" for i in range(n_feats)]
    importance = pd.DataFrame({"feature": feat_names, "mean_abs_shap": np.mean(np.abs(shap_values), axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    out_tab = Path(cfg["outputs"]["tables"]) / "shap_importance.csv"
    importance.to_csv(out_tab, index=False)

    # summary plot
    os.makedirs(cfg["outputs"]["figures"], exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X_trans, show=False)
    plt.tight_layout()
    plt.savefig(Path(cfg["outputs"]["figures"]) / "shap_summary.png")
    print("SHAP analysis complete")


if __name__ == "__main__":
    import numpy as np
    main()
