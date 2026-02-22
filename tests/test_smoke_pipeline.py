import subprocess
import shutil

def test_smoke_run_all():
    scripts = [
        "analysis/01_data_qc.py",
        "analysis/02_preprocess.py",
        "analysis/03_icc_hlm.R",
        "analysis/04_holdout_split.py",
        "analysis/05_nested_groupkfold_models.py",
        "analysis/06_model_comparison.py",
        "analysis/07_interpretability_shap.py",
        "analysis/08_ale_analysis.py",
        "analysis/09_bootstrap_stability.py",
        "analysis/10_tables_figures.py",
        "analysis/99_environment_report.py",
    ]
    import sys
    for script in scripts:
        if script.endswith('.R'):
            # skip if Rscript not installed
            if shutil.which("Rscript") is None:
                continue
            cmd = ["Rscript", script]
        else:
            cmd = [sys.executable, script]
        subprocess.run(cmd, check=True)
