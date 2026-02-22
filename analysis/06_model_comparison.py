import os
from pathlib import Path

import pandas as pd
import yaml


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    in_path = Path(cfg["outputs"]["model_artifacts"]) / "nested_cv_results.csv"
    df = pd.read_csv(in_path)
    summary = df.agg({"r2": ["mean", "std"], "mse": ["mean", "std"]})
    out_dir = Path(cfg["outputs"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.csv"
    summary.to_csv(out_path)
    print(f"Model comparison table written to {out_path}")


if __name__ == "__main__":
    main()
