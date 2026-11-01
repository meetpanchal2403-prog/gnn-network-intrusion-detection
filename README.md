# Multimodal GNN-Based Network Intrusion Detection

Research implementation for host-level, temporally windowed graph-based
network intrusion detection using network-flow and system-log features.

## Evaluation setup

- Seven scenarios: fox, harrison, russellmitchell, santos, wardbeck, wheeler, and wilson.
- Leave-one-scenario-out (LOSO) evaluation.
- Five-minute windows.
- Three model seeds per scenario in the primary prediction artifact.
- Multimodal graph model using flow features, system-log features, and log-message embeddings.

The shaw scenario is excluded from the current seven-fold artifacts.

## Current audit status

The repository is being consolidated from existing experiments. Fold assignments and scaler checks have been inspected, but this does not prove that every preprocessing or feature-leakage risk has been eliminated. Clean-clone reproduction of the entire pipeline has not yet been established.

A period-classification audit identified one Wardbeck window (1307) containing escalation-related labels that was originally assigned to the clean period. The five CHAIN_LABELS definitions in the notebook have been updated to include those labels.

The correction audit is documented in:

- `docs/period_classification_correction_audit.md`
- `results/tables/edge_multi_period_metrics_corrected.csv`

The corresponding corrected prediction file,
`data/ait/processed/revision/pred_edge_multi_periods_corrected.csv`,
is not included in this repository. It is part of the local processed-data
artifacts and must not be assumed to be available in a clean clone.

The original prediction CSV and saved fold files were not overwritten. No retraining was performed as part of the period-classification correction. Corrected metrics are an audit output, not a freshly retrained model result.

## Data access and configuration

The processed-data directory is configured in `configs/config.yaml`.
The default path is `data/ait/processed`, relative to the repository root.
Place the required processed files there, or change this setting to the
location used in your environment.

Large datasets, pickle files, and model checkpoints are not intended to be
committed to Git. Obtain source data through the appropriate dataset records
and follow their access and licensing terms.

## Environment

Use a compatible Python environment. Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

See [`docs/environment.md`](docs/environment.md) for installation guidance and
recorded environment observations. PyTorch and PyTorch Geometric require
platform-specific installation steps. Exact dependency versions and
clean-clone reproduction still need verification.

## Repository contents

```text
.
├── analysis/
│   └── full_capstone.ipynb
├── configs/
│   └── config.yaml
├── docs/
│   ├── fold_validation_summary.json
│   ├── loso_fold_assignments.json
│   ├── model_pipeline_audit.md
│   ├── notebook_audit.md
│   └── period_classification_correction_audit.md
├── results/
│   └── tables/
├── .gitignore
├── audit_model_pipeline.py
├── audit_notebook.py
├── audit_saved_predictions.py
├── correct_prediction_periods.py
├── generate_fold_metadata.py
├── recalculate_corrected_metrics.py
├── requirements.txt
└── README.md
```

This tree describes the files currently tracked in the repository; it does
not imply that the full training pipeline or all required datasets are
included.

## Citing this work

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Once a version is archived on Zenodo, cite the DOI for the exact archived release.

## Results and reproducibility

Historical experiment outputs should be distinguished from corrected audit
outputs and from results reproduced from a verified final implementation.
Do not claim that the entire pipeline is reproducible until its required data,
dependencies, preprocessing steps, and evaluation procedure have been
verified from a clean environment.
