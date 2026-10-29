import json
from pathlib import Path

repo = Path(__file__).resolve().parent
report_path = Path.home() / "capstone_fold_validation.json"

report = json.loads(report_path.read_text(encoding="utf-8"))

assignments = {}
for fold in report["folds"]:
    test = fold["test_scenario"]
    assignments[test] = {
        "test_scenario": test,
        "validation_scenario": fold["val_scenario"],
        "train_scenarios": fold["train_scenarios"],
        "window_minutes": fold["window_minutes"],
        "source_file": fold["file"],
        "structural_checks": fold["checks"]
    }

(repo / "docs" / "loso_fold_assignments.json").write_text(
    json.dumps(assignments, indent=2), encoding="utf-8"
)

summary = {
    "fold_files_found": report["checks"]["fold_files_found"],
    "folds_inspected": report["checks"]["folds_inspected"],
    "all_seven_unique_test_scenarios":
        report["checks"]["all_seven_unique_test_scenarios"],
    "all_fold_checks_pass": report["checks"]["all_fold_checks_pass"],
    "errors": report["checks"]["errors"],
    "note": "Structural checks only; preprocessing leakage is not assessed."
}
(repo / "docs" / "fold_validation_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("Generated LOSO assignment metadata.")
print(json.dumps(summary, indent=2))
