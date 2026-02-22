import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["raw"])
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.get("holdout_fraction", 0.2), random_state=cfg.get("random_seed"))
    groups = df[cfg["group_var"]]
    train_idx, hold_idx = next(gss.split(df, groups=groups))
    df["split"] = "train"
    df.loc[hold_idx, "split"] = "holdout"
    out_dir = Path(cfg["outputs"]["model_artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "holdout_split.csv"
    df[["id", "split", cfg["group_var"]]].to_csv(out_path, index=False)
    print(f"Saved holdout split to {out_path}")


if __name__ == "__main__":
    main()
