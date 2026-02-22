import platform
import sys
from pathlib import Path

import importlib.metadata as metadata
import yaml


def load_config():
    cfg_path = Path(__file__).parent / "00_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    out_dir = Path(cfg["outputs"]["logs"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "environment.txt"
    with open(out_path, "w") as f:
        f.write(f"Python version: {platform.python_version()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'] if 'Name' in d.metadata else ''):
            name = dist.metadata.get('Name', dist.metadata.get('name', ''))
            version = dist.version
            f.write(f"{name}=={version}\n")
    print(f"Environment report written to {out_path}")


if __name__ == "__main__":
    main()
