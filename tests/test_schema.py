import json
import yaml
import pandas as pd
from pathlib import Path

def test_schema_matches_data():
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "analysis/00_config.yaml"))
    df = pd.read_csv(cfg["data"]["raw"])
    schema = json.load(open(cfg["data"]["schema"]))
    expected = [v["name"] for v in schema.get("variables", [])]
    assert set(expected) == set(df.columns), "Data columns differ from schema"
