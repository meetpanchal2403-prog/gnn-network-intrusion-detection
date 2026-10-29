import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "ait" / "processed"

FOLD_PATH = DATA_ROOT / "loso_fold_test-wardbeck_w5m.pkl"
PRED_PATH = DATA_ROOT / "revision" / "pred_edge_multi.csv"
OUT_PATH = DATA_ROOT / "revision" / "pred_edge_multi_periods_corrected.csv"

EXTRA_CHAIN_LABELS = {
    "escalate",
    "attacker_change_user",
    "escalated_command",
    "escalated_sudo_command",
    "escalated_sudo_session",
}

EXFIL_LABELS = {
    "data exfiltration", "attacker", "dnsteal", "dnsteal-received"
}

# Preserve the existing period-classification rules.
CHAIN_LABELS = {
    "service_scan", "dirb_scan", "wpscan", "dns_brute_force_start",
    "host_discover_dmz", "host_discover_local", "online_cracking",
    "upload_rce_shell", "dump_wp_users", "check_whoami",
    "check_uname_r", "read_profile", "check_who", "check_meminfo",
    "check_uname_a", "check_user_id", "check_df", "check_netstat_nat",
    "check_id", "read_resolv", "check_netstat_t", "list_home",
    "check_last", "list_web_dir", "check_date", "list_www",
    "check_netstat_l", "clear", "check_wp_config", "check_ps_a",
    "read_passwd", "check_release", "check_cpuinfo", "read_group",
    "check_uptime", "check_network_config",
    "update/command on unassigned port", "check_pwd", "list_l",
    "vpn_connect",
}

def normalize_labels(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return set(map(str, value))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {x.strip() for x in str(value).split("|") if x.strip()}

def build_block_ids(periods, scenario):
    block_ids = np.empty(len(periods), dtype=object)
    current = None
    block_num = -1
    for i, period in enumerate(periods):
        if period != current:
            block_num += 1
            current = period
        block_ids[i] = f"{scenario}_blk{block_num}_{current}"
    return block_ids

if not FOLD_PATH.exists():
    raise FileNotFoundError(f"Fold not found: {FOLD_PATH}")
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Predictions not found: {PRED_PATH}")

with FOLD_PATH.open("rb") as f:
    fold = pickle.load(f)

scenario = fold["scenarios"]["wardbeck"]
label_table = scenario["label_table"].copy()
window_ids = np.asarray(scenario["window_ids"], dtype=int)
old_periods = np.asarray(scenario["window_periods"], dtype=object)

if len(window_ids) != len(old_periods):
    raise ValueError("Fold window_ids and window_periods have different lengths.")

# Reclassify windows using the label table; chain takes priority over exfil.
labels_by_window = {}
for row in label_table.itertuples(index=False):
    wid = int(row.window_id)
    labels_by_window.setdefault(wid, set()).update(
        normalize_labels(getattr(row, "labels", []))
    )

new_periods = []
for wid, old_period in zip(window_ids, old_periods):
    labels = labels_by_window.get(int(wid), set())
    if labels & (CHAIN_LABELS | EXTRA_CHAIN_LABELS):
        period = "chain"
    elif labels & EXFIL_LABELS:
        period = "exfil"
    else:
        period = "clean"
    new_periods.append(period)

new_periods = np.asarray(new_periods, dtype=object)
changed = [
    (int(w), str(old), str(new))
    for w, old, new in zip(window_ids, old_periods, new_periods)
    if old != new
]

print("Changed fold windows:", changed)
if changed != [(1307, "clean", "chain")]:
    raise RuntimeError(
        "Unexpected period changes. No output written. Review the labels/rules above."
    )

# Rebuild blocks in the saved fold's original window sequence.
new_block_ids = build_block_ids(new_periods, "wardbeck")
period_by_window = dict(zip(window_ids, new_periods))
block_by_window = dict(zip(window_ids, new_block_ids))

pred = pd.read_csv(PRED_PATH)
required = {"test_scenario", "window_id", "period", "block_id", "prob", "y_true", "seed"}
missing = required - set(pred.columns)
if missing:
    raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

corrected = pred.copy()
mask = corrected["test_scenario"].eq("wardbeck")
corrected.loc[mask, "period"] = corrected.loc[mask, "window_id"].map(period_by_window)
corrected.loc[mask, "block_id"] = corrected.loc[mask, "window_id"].map(block_by_window)

if corrected.loc[mask, "period"].isna().any() or corrected.loc[mask, "block_id"].isna().any():
    raise RuntimeError("Some Wardbeck prediction windows did not match the saved fold.")

# Ensure predictions, labels, probabilities, seeds and non-Wardbeck rows did not change.
unchanged_cols = [c for c in pred.columns if c not in {"period", "block_id"}]
pd.testing.assert_frame_equal(
    pred[unchanged_cols], corrected[unchanged_cols], check_dtype=False
)
pd.testing.assert_frame_equal(
    pred.loc[~mask].reset_index(drop=True),
    corrected.loc[~mask].reset_index(drop=True),
    check_dtype=False
)

corrected.to_csv(OUT_PATH, index=False)

print(f"Original rows: {len(pred):,}")
print(f"Corrected output: {OUT_PATH}")
print("Original prediction file was not overwritten.")
print("Changed Wardbeck rows:", int((mask & pred["window_id"].eq(1307)).sum()))
print("Wardbeck period counts after correction:")
print(corrected.loc[mask].groupby("period").size().to_string())
print("Validation passed: probabilities, labels, seeds and other scenarios unchanged.")
