import pandas as pd
import subprocess
from pathlib import Path

def test_group_split_no_leakage():
    # run holdout split script with current interpreter
    import sys
    subprocess.run([sys.executable, "analysis/04_holdout_split.py"], check=True)
    # load config and split output
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "analysis/00_config.yaml"))
    split = pd.read_csv(Path(cfg["outputs"]["model_artifacts"]) / "holdout_split.csv")
    raw = pd.read_csv(cfg["data"]["raw"])
    # ensure no group appears in both train and holdout
    groups_train = set(split.loc[split.split == "train", cfg["group_var"]])
    groups_hold = set(split.loc[split.split == "holdout", cfg["group_var"]])
    assert groups_train.isdisjoint(groups_hold), "Group leakage between train and holdout"
