from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    matthews_corrcoef,
)

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data" / "ait" / "processed"

PRED_PATH = DATA_ROOT / "revision" / "pred_edge_multi_periods_corrected.csv"
OUT_PATH = REPO_ROOT / "results" / "tables" / "edge_multi_period_metrics_corrected.csv"

THRESHOLD = 0.5
N_BOOT = 500

def metric(yt, yp):
    # Match the notebook's special convention for an empty positive class.
    if yt.sum() == 0 and yp.sum() == 0:
        return 1.0, 1.0, 1.0
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)

def block_bootstrap_ci(yt, yp, block_ids, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_ids)
    block_to_idx = {
        b: np.flatnonzero(block_ids == b) for b in unique_blocks
    }
    f1s = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        idx = np.concatenate([block_to_idx[b] for b in sampled])
        f1s.append(metric(yt[idx], yp[idx])[2])
    return (
        float(np.mean(f1s)),
        float(np.percentile(f1s, 2.5)),
        float(np.percentile(f1s, 97.5)),
    )

if not PRED_PATH.exists():
    raise FileNotFoundError(f"Corrected predictions not found: {PRED_PATH}")

df = pd.read_csv(PRED_PATH)
required = {
    "test_scenario", "window_id", "period", "block_id",
    "y_true", "prob", "seed", "model"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["y_true"] = df["y_true"].astype(int)
df["prob"] = df["prob"].astype(float)
df["window_id"] = df["window_id"].astype(int)

rows = []
for (scenario, seed), group in df.groupby(["test_scenario", "seed"], sort=True):
    for period in ["exfil", "chain", "clean"]:
        m = group[group["period"] == period].copy()
        if m.empty:
            continue

        yt = m["y_true"].to_numpy(dtype=int)
        probs = m["prob"].to_numpy(dtype=float)
        yp = (probs > THRESHOLD).astype(int)

        p, r, f1 = metric(yt, yp)

        # Match notebook: block ID combined with a six-window chunk.
        bids = (
            m["block_id"].astype(str) + "_" +
            (m["window_id"] // 6).astype(str)
        ).to_numpy()

        _, lo, hi = block_bootstrap_ci(
            yt, yp, bids, n_boot=N_BOOT,
            seed=seed * 1000 + ({"exfil": 1, "chain": 2, "clean": 3}[period])
        )

        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        n_pos = int(yt.sum())
        n_neg = int((yt == 0).sum())

        auprc = (
            float(average_precision_score(yt, probs))
            if 0 < n_pos < len(yt) else np.nan
        )
        mcc = (
            float(matthews_corrcoef(yt, yp))
            if 0 < n_pos < len(yt) else np.nan
        )

        rows.append({
            "test_scenario": scenario,
            "period": period,
            "seed": seed,
            "model": str(m["model"].iloc[0]),
            "n": len(m),
            "n_pos": n_pos,
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
            "f1_ci_low": round(lo, 6),
            "f1_ci_high": round(hi, 6),
            "false_positives": fp,
            "false_positive_rate": round(fp / max(n_neg, 1), 6),
            "auprc": round(auprc, 6) if np.isfinite(auprc) else np.nan,
            "mcc": round(mcc, 6) if np.isfinite(mcc) else np.nan,
        })

out = pd.DataFrame(rows)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"Saved corrected metrics: {OUT_PATH}")
print(f"Metric rows: {len(out)}")
print("\nMean F1 by period (averaged across scenario-seed rows):")
print(out.groupby("period")["f1"].agg(["mean", "min", "max"]).round(4))
print("\nWardbeck metrics:")
print(
    out[out["test_scenario"] == "wardbeck"][
        ["test_scenario", "seed", "period", "n_pos", "precision",
         "recall", "f1", "f1_ci_low", "f1_ci_high", "false_positive_rate"]
    ].to_string(index=False)
)
