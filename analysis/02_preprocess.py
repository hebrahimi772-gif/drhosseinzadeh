import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config():
    config_path = Path(__file__).parent / "00_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(df, cfg):
    # infer numeric vs categorical
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # drop id, group, target if present
    for drop in ["id", "org_id_anon", cfg.get("target")]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    transformer = ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop"
    )
    pipeline = Pipeline([("transformer", transformer)])
    return pipeline


def main():
    global cfg
    cfg = load_config()
    data_path = cfg["data"]["raw"]
    df = pd.read_csv(data_path)

    pipeline = build_pipeline(df, cfg)

    os.makedirs(cfg["outputs"]["model_artifacts"], exist_ok=True)
    out_path = Path(cfg["outputs"]["model_artifacts"]) / "preprocessing_pipeline.pkl"
    joblib.dump(pipeline, out_path)
    print(f"Pipeline saved to {out_path}")


if __name__ == "__main__":
    main()
