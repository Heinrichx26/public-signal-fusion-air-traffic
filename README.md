# Asynchronous Credal Residual Fusion for Operational State Inference in Air Traffic Disruptions

This repository contains code, source-data instructions, field schemas, split definitions, baseline scores, and selected result tables for:

**Asynchronous Credal Residual Fusion for Operational State Inference in Air Traffic Disruptions**

## Data sources

The raw sources used by the study are accessible from official data providers:

- Bureau of Transportation Statistics Airline On-Time Performance data.
- Iowa Environmental Mesonet Automated Surface Observing Systems reports.
- Federal Aviation Administration Air Traffic Control System Command Center advisory archive.
- National Centers for Environmental Information Storm Events Database.

The repository does not redistribute large raw source files. The `manifests/` directory records source files and download status used in the study. The scripts in `src/data/` document the acquisition workflow.

## Repository layout

- `src/data/`: source acquisition and parsing scripts.
- `src/analysis/`: airport-hour panel construction, asynchronous credal residual fusion (ACRF), residual score, advisory-outcome hysteresis, event validation, and robustness scripts.
- `src/plotting/`: figure builders that read result tables.
- `results/benchmark/`: benchmark field dictionary, tasks, split definitions, and baseline scores.
- `results/scorecards/`: selected result tables.
- `manifests/`: source-data manifests.

## Quick reproducibility path

Install the Python environment:

```bash
python -m pip install -r requirements.txt
```

Verify the package:

```bash
python src/analysis/verify_release_package.py
```

This command checks the README, environment file, data schema, task definitions, split identifiers, baseline scores, and selected scorecards. It writes `results/release_package_audit.csv` and `results/release_table_summary.csv`.

Run the smoke path before full reconstruction:

```bash
python src/analysis/acrf_smoke_test.py --months 1,7,12 --airports ATL,ORD --output-name acrf_smoke_test
python src/analysis/fusion_framework_strengthening_smoke.py --months 1,7,12 --output-name demand_residual_smoke
python src/analysis/fusion_strengthening_prediction_diagnostics.py --output-name prediction_diagnostics_smoke --airports ATL,ORD
```

After reconstructing the full result tree from the raw sources, rebuild the benchmark tables and diagnostics with:

```bash
python src/analysis/acrf_smoke_test.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name acrf_full_2025_eta_auc_grid4 --eta-objective auc --eta-grid 0,0.5,1,1.5,2,2.5,3,3.5,4
python src/analysis/fusion_strengthening_demand_residual.py --months 1-12 --output-name demand_residual_full_2025
python src/analysis/fusion_strengthening_prediction_diagnostics.py --output-name prediction_diagnostics_full --airports ALL
python src/analysis/fusion_strengthening_hysteresis.py
python src/analysis/fusion_strengthening_benchmark_package.py
```

Rebuild benchmark figures with:

```bash
python src/plotting/build_reported_figures.py
python src/plotting/build_strengthening_figures.py
```

To reconstruct raw-source inputs, use the source manifests in `manifests/` together with the acquisition scripts in `src/data/`. The large raw-source files are not redistributed in this repository.

## Evaluation tasks

The package defines five evaluation tasks:

1. Residual state detection.
2. Long-delay prediction.
3. Cancellation prediction.
4. Post-advisory persistence.
5. Event-level advisory validation.

The package includes fixed split definitions for leave-one-month, 2024-to-2025, leave-one-airport, and event bootstrap evaluation. Selected scorecards include ACRF model metrics, learned eta selection, residual-belief separation, Advisory-Outcome Hysteresis Index (AOHI), PR-AUC and calibration diagnostics, leave-one-airport stability, event pre-trend contrasts, residual-score lift, and source-reliability ranking.

## Reproducibility note

The plotting scripts read saved result tables and do not rerun full experiments. Full data reconstruction requires downloading source files listed in the manifests.

## Citation

Please cite the associated publication after publication if you use this code or the derived tables.
