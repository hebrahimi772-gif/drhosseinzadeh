import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config():
    config_path = Path(__file__).parent / "00_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main():
    cfg = load_config()
    os.makedirs(cfg["outputs"]["logs"], exist_ok=True)
    log_file = Path(cfg["outputs"]["logs"]) / "01_data_qc.log"
    setup_logging(log_file)

    data_path = cfg["data"]["raw"]
    schema_path = cfg["data"]["schema"]

    df = pd.read_csv(data_path)
    logging.info(f"Read {len(df)} records from {data_path}")

    with open(schema_path) as f:
        schema = json.load(f)
    expected_cols = [v["name"] for v in schema.get("variables", [])]
    missing_cols = set(expected_cols) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_cols)
    if missing_cols:
        logging.error(f"Missing columns in data: {missing_cols}")
        raise ValueError("Schema mismatch")
    logging.info("Column names match schema")

    # missingness and type checks
    summary = []
    for col in df.columns:
        count_missing = df[col].isna().sum()
        dtype = str(df[col].dtype)
        summary.append({"variable": col, "dtype": dtype, "missing": count_missing})
    summary_df = pd.DataFrame(summary)
    os.makedirs(cfg["outputs"]["tables"], exist_ok=True)
    out_path = Path(cfg["outputs"]["tables"]) / "qc_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logging.info(f"Wrote QC summary to {out_path}")


if __name__ == "__main__":
    main()
