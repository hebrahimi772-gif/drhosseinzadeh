# Predictive Modeling of Organizational Change Management Capability in Healthcare Settings: An Integrated Machine Learning and Hierarchical Statistical Approach

This repository accompanies a manuscript evaluating methods for predicting organizational change management capability using a combination of machine learning techniques and hierarchical statistical modeling. The package provides all materials necessary for a journal reviewer to reproduce the analyses and generate the tables and figures presented in the paper.

## Repository Purpose

The primary goal of this repository is to offer a fully executable environment where the data cleaning, modeling, evaluation, and interpretation steps described in the manuscript can be rerun using only the included files. All code is deterministic and configured via a central YAML file. Data artifacts are fixed and versioned for transparency.

## Contents Overview

The repository is structured as follows:

- `data/` – raw datasets and metadata (schema, manifest, codebook).
- `analysis/` – stepwise scripts implementing the workflow from quality control through modeling and output generation.
- `outputs/` – directories for tables, figures, logs, and model artifacts produced by the analysis.
- `env/` – requirements and optional Dockerfile for environment setup.
- `docs/` – narrative documentation covering reproducibility scope, methods protocol, data dictionary, and ethics.
- `tests/` – automated checks and smoke tests to ensure structural integrity and prevent data leakage.
- `.github/workflows/ci.yml` – continuous integration configuration for running tests on push and pull requests.

## Reproducibility Scope

This repository is public and contains the complete dataset (`data/raw/ocmcap_raw_dataset_v1.csv`), associated schema (`data/raw/schema_ocmcap_v1.json`), manifest (`data/metadata/dataset_manifest_ocmcap_v1.csv`), and codebook (`data/metadata/codebook_ocmcap_v1.csv`). All analytical code needed to reproduce the results reported in the manuscript is included. The dataset has been anonymized and contains no direct identifiers.  The execution of the analyses is deterministic given the fixed random seed in `analysis/00_config.yaml`. Group-aware validation procedures are enforced to mirror the design in the paper. No external data sources are required.

## Data Section

The following files are provided for the dataset and metadata:

- `data/raw/ocmcap_raw_dataset_v1.csv` – primary anonymized survey responses (230 records, 119 variables).
- `data/raw/schema_ocmcap_v1.json` – JSON schema describing variable types and dataset characteristics.
- `data/metadata/dataset_manifest_ocmcap_v1.csv` – high‑level manifest of dataset provenance and usage restrictions.
- `data/metadata/codebook_ocmcap_v1.csv` – detailed codebook listing variable names, labels, types, and valid ranges.

These files should remain unchanged to ensure reproducibility.

## Quickstart

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements.txt
```

If you prefer containerized execution, a minimal `env/Dockerfile` is provided; see its comments for build instructions.

### Running the analysis

All scripts read from `analysis/00_config.yaml` and produce outputs under `outputs/`. From the repository root:

```bash
bash -c "
python analysis/01_data_qc.py &&
python analysis/02_preprocess.py &&
python analysis/03_icc_hlm.R &&
python analysis/04_holdout_split.py &&
python analysis/05_nested_groupkfold_models.py &&
python analysis/06_model_comparison.py &&
python analysis/07_interpretability_shap.py &&
python analysis/08_ale_analysis.py &&
python analysis/09_bootstrap_stability.py &&
python analysis/10_tables_figures.py &&
python analysis/99_environment_report.py
"
```

Alternatively, run individual scripts as needed.

## Reproduce Manuscript Outputs

Each analysis script corresponds to a section in the manuscript:

1. `01_data_qc.py` – quality control summary.
2. `02_preprocess.py` – feature engineering and pipeline definitions.
3. `03_icc_hlm.R` – intraclass correlation/hierarchical model.
4. `04_holdout_split.py` – hold‑out dataset creation.
5. `05_nested_groupkfold_models.py` – nested cross-validated model training.
6. `06_model_comparison.py` – metric aggregation and comparison table.
7. `07_interpretability_shap.py` – SHAP interpretability.
8. `08_ale_analysis.py` – ALE plots for top features.
9. `09_bootstrap_stability.py` – stability assessment via bootstrapping.
10. `10_tables_figures.py` – compile final tables and figures.
11. `99_environment_report.py` – log environment details.

run the full sequence using the command above.

## Computing Environment

Analyses were developed and tested on Python 3.11. Key packages include `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `scipy`, and `statsmodels`. Versions are pinned in `env/requirements.txt`. R scripts (used for ICC/HLM) rely on `lme4` (and `yaml` for configuration); an `env/renv.lock` file provides minimal package specifications. Users may install required R packages via `install.packages(c("lme4","yaml"))`.

## Citation

Please cite the associated manuscript when using this repository. See `CITATION.cff` for the recommended citation format.

## License

Code is released under the MIT License; data and documentation are shared under a CC-BY‑NC‑4.0 equivalent (see LICENSE file for details).

## Contact

Corresponding author: hebrahimi772-gif