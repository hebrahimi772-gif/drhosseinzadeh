# Methods Protocol

This document outlines the step-by-step analytical protocol and links each stage to the corresponding script and configurable parameters.

1. **Data Quality Control** (`analysis/01_data_qc.py`)
   - Reads raw CSV using paths from `00_config.yaml`.
   - Validates that column names match the JSON schema and checks types.
   - Computes missingness by variable and range violations against the schema.
   - Outputs a summary CSV in `outputs/tables/`.
   - Reviewers may adjust quality thresholds or include additional checks via the script parameters.

2. **Preprocessing** (`analysis/02_preprocess.py`)
   - Constructs an `sklearn` `ColumnTransformer` pipeline based on variable types (numeric scaling, one-hot encoding for categoricals).
   - The pipeline is not fit until invoked within cross-validation to avoid data leakage; the script saves the pipeline object to `outputs/model_artifacts/`.
   - Feature selection and imputation strategies are defined here and can be adjusted in `00_config.yaml` under preprocessing settings.

3. **ICC / Hierarchical Step** (`analysis/03_icc_hlm.R`)
   - Uses `lme4::lmer` to fit a null random-intercept model of the target over `org_id_anon`.
   - Computes the intraclass correlation coefficient (ICC) and writes `outputs/tables/icc_summary.csv`.
   - The group variable name can be edited via the config file when invoking the R script with `--config` if needed.

4. **Hold-out Split** (`analysis/04_holdout_split.py`)
   - Performs a stratified, group-aware split of the dataset into training and hold-out sets using `GroupKFold` with a single fold.
   - The holdout fraction is set in `00_config.yaml` (`holdout_fraction`).
   - Generated indices are saved to `outputs/model_artifacts/holdout_split.csv`.

5. **Nested Group-aware Cross-validation** (`analysis/05_nested_groupkfold_models.py`)
   - Loads training portion and executes outer `GroupKFold` folds defined by `outer_folds` in the config.
   - Within each outer fold, an inner `GroupKFold` loop is used for hyperparameter tuning of specified models (e.g. `RandomForestRegressor` with grid search).
   - Metrics defined in the config (`scoring_metrics`) are computed and stored in a fold-level CSV under `outputs/model_artifacts/`.
   - The primary model type and hyperparameter grids are controlled via the configuration file.

6. **Model Comparison Metrics** (`analysis/06_model_comparison.py`)
   - Aggregates fold results, computes average performance and bootstrapped confidence intervals.
   - Saves a comparison table in `outputs/tables/model_comparison.csv`.

7. **Interpretability (SHAP)** (`analysis/07_interpretability_shap.py`)
   - Trains the chosen final model on the full training data (as per hold-out split) using the preprocessing pipeline.
   - Computes SHAP values via the `shap` library.
   - Outputs a summary importance CSV to `outputs/tables/` and a SHAP summary plot PNG to `outputs/figures/`.

8. **Accumulated Local Effect (ALE) Analysis** (`analysis/08_ale_analysis.py`)
   - For the top-ranked features from SHAP, computes ALE plots using `alibi` or custom code.
   - Saves PNGs to `outputs/figures/`.

9. **Bootstrap Stability** (`analysis/09_bootstrap_stability.py`)
   - Resamples the training data with group-aware bootstrap (sampling organizations with replacement).
   - Retrains the model on each bootstrap sample and records coefficient or feature importance variability.
   - Outputs a stability statistics table to `outputs/tables/`.

10. **Tables and Figures Assembly** (`analysis/10_tables_figures.py`)
    - Collects all intermediate outputs and renames/moves them to final manuscript names (e.g. `Table_1.csv`, `Figure_1.png`).
    - Places final artifacts under `outputs/tables/` and `outputs/figures/`.

11. **Environment Report** (`analysis/99_environment_report.py`)
    - Prints Python and package versions; writes them to `outputs/logs/environment.txt`.

Reviewers may adjust parameters (e.g. fold counts, random seed) in `analysis/00_config.yaml` before rerunning the workflow.