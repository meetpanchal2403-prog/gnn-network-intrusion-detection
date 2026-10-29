# Period Classification Correction Audit

Date: 2026-10-29

## Purpose

A review of the LOSO evaluation identified one Wardbeck window whose
period classification did not include escalation-related labels in the
attack-chain label set.

## Correction

Wardbeck window 1307 contains the labels:

- attacker_change_user
- escalate
- escalated_command
- escalated_sudo_command
- escalated_sudo_session

This window was classified as `clean` in the original saved fold. The
corrected audit classifies it as `chain`, preserving the existing rule
that chain takes priority over exfiltration.

The comparison found no period changes in fox, harrison, russellmitchell,
santos, wheeler, or wilson.

## Outputs

- Original predictions: `data/ait/processed/revision/pred_edge_multi.csv`
- Corrected predictions: `data/ait/processed/revision/pred_edge_multi_periods_corrected.csv`
- Corrected metrics: `results/tables/edge_multi_period_metrics_corrected.csv`
- Corrected metrics using the notebook metric convention:
  `results/tables/edge_multi_period_metrics_corrected_notebook_convention.csv`

The original prediction CSV and saved fold files were not overwritten.
No model retraining was performed. The correction changes evaluation
period assignments, not model predictions or probabilities.

## Corrected Summary

Mean F1 across scenario-seed rows:

| Period | Mean F1 | Minimum F1 | Maximum F1 |
|---|---:|---:|---:|
| Exfil | 0.9944 | 0.9601 | 1.0000 |
| Chain | 0.7087 | 0.2927 | 0.9259 |
| Clean | 0.1905 | 0.0000 | 1.0000 |

These are averages across scenario-seed rows, not pooled predictions.

## Metric and Bootstrap Notes

The metric function follows the notebook convention: if both the actual
and predicted positive counts are zero, precision, recall, and F1 are
set to 1. Otherwise, standard binary metrics are used with
`zero_division=0`.

Clean-period F1 must be interpreted cautiously because clean periods
may contain no actual positive examples. False-positive rate should
also be reported.

Confidence intervals in the corrected metrics files use 500 block
bootstrap replicates. Blocks are formed from the corrected period block
IDs combined with six-window chunks. These are separate audit outputs
and should not silently replace the original experiment results.

## Scope and Limitations

This audit corrects the evaluation-period assignment for the identified
window. It does not retrain the model, change probabilities, or establish
that every possible label or feature-leakage issue has been ruled out.

The notebook's period-classification definitions and saved fold metadata
have not been globally rewritten by this audit. The correction is applied
to the separate corrected prediction and metrics outputs.
