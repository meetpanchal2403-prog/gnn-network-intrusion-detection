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

Separate corrected audit outputs are available:

- `data/ait/processed/revision/pred_edge_multi_periods_corrected.csv`
- `results/tables/edge_multi_period_metrics_corrected.csv`
- `docs/period_classification_correction_audit.md`

The original prediction CSV and saved fold files were not overwritten. No retraining was performed as part of the period-classification correction. Corrected metrics are an audit output, not a freshly retrained model result.

## Data access

Processed artifacts are expected at the path configured in `configs/config.yaml`. Large datasets, pickle files, and model checkpoints are not intended to be committed to Git. Obtain source data through the appropriate dataset records and follow access and licensing terms.

## Environment

Use a compatible Python environment. Install the listed dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may require platform-specific installation steps. Exact dependency versions and clean-clone reproduction still need verification against the original working environment.

## Repository layout

- `data_prep/`: data validation and preprocessing
- `models/`: model implementation
- `training/`: training and evaluation code
- `analysis/`: notebook and audit utilities
- `configs/`: experiment configuration
- `tests/`: validation tests
- `docs/`: audit reports and documentation
- `results/`: curated metrics, tables, and figures

## Results and reproducibility

Historical experiment outputs should be distinguished from corrected audit outputs and from results reproduced from a verified final implementation. Do not claim that the entire pipeline is reproducible until its required data, dependencies, preprocessing steps, and evaluation procedure have been verified from a clean environment.
