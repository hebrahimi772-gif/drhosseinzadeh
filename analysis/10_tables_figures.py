import os
import shutil
from pathlib import Path

import yaml


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    tdir = Path(cfg["outputs"]["tables"])
    fdir = Path(cfg["outputs"]["figures"])
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "qc_summary.csv": "Table_1.csv",
        "model_comparison.csv": "Table_2.csv",
        "shap_importance.csv": "Table_3.csv",
        "bootstrap_stability.csv": "Table_4.csv",
    }
    for src, dst in mapping.items():
        src_path = tdir / src
        dst_path = tdir / dst
        if src_path.exists():
            shutil.copy(src_path, dst_path)
    # figures
    figmap = {
        "shap_summary.png": "Figure_1.png",
    }
    for src, dst in figmap.items():
        src_path = fdir / src
        dst_path = fdir / dst
        if src_path.exists():
            shutil.copy(src_path, dst_path)
    print("Tables and figures organized")


if __name__ == "__main__":
    main()
