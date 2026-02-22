# Reproducibility Scope

This document clarifies what a reviewer can expect when using the public materials in this repository.

## Public Assets

- The full analysis code base resides in the `analysis/` directory and is open-source under the MIT license.
- Data artifacts include:
  - `data/raw/ocmcap_raw_dataset_v1.csv` (primary survey responses)
  - `data/raw/schema_ocmcap_v1.json` (variable definitions, primary key, group key)
  - `data/metadata/dataset_manifest_ocmcap_v1.csv` (dataset provenance)
  - `data/metadata/codebook_ocmcap_v1.csv` (variable labels, types, ranges)

All of the above may be inspected and executed without restriction. No additional data are required.

## Execution Expectations

- The scripts are designed to run deterministically; a single `random_seed` set in `analysis/00_config.yaml` controls all stochastic procedures.
- Group-aware procedures are enforced throughout: the script `04_holdout_split.py` creates a hold-out set that does not share `org_id_anon` values with the training data, and cross-validation in `05_nested_groupkfold_models.py` uses `GroupKFold` in the outer loop. This mirrors the design described in the manuscript.
- Any reviewer wishing to alter these settings may modify `00_config.yaml` prior to execution.

## Limitations and Anonymization

- Organizational identifiers (`org_id_anon`) have been anonymized as integers 1–15. No real organization names or other identifiers are present.
- Individual-level records are pseudonymous and contain no direct personal identifiers.
- While the provided dataset is complete for the purposes of the paper, users should be aware that it represents a survey-based convenience sample from healthcare settings; generalization beyond the observed units is not guaranteed.

By following the instructions in the repository, a reviewer should be able to replicate all reported tables and figures exactly using the supplied materials.