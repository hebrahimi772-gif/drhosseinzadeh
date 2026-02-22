import subprocess
import joblib
import yaml
from pathlib import Path


def test_preprocessing_unfitted():
    import sys
    subprocess.run([sys.executable, "analysis/02_preprocess.py"], check=True)
    cfg = yaml.safe_load(open(Path(__file__).parent.parent / "analysis/00_config.yaml"))
    pipe_path = Path(cfg["outputs"]["model_artifacts"]) / "preprocessing_pipeline.pkl"
    pipe = joblib.load(pipe_path)
    assert not hasattr(pipe, "n_features_in_"), "Preprocessing pipeline appears to be fitted on full data"
