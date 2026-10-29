from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent

src = PROJECT_ROOT / "data" / "ait" / "processed" / "revision" / "pred_edge_multi.csv"
out = REPO_ROOT / "results" / "tables" / "edge_multi_metric_audit.csv"

df = pd.read_csv(src)
df["y_true"] = df["y_true"].astype(int)
df["prob"] = df["prob"].astype(float)
df["pred"] = (df["prob"] > 0.5).astype(int)

rows = []

for (sc, seed, period), g in df.groupby(
    ["test_scenario", "seed", "period"], dropna=False
):
    yt = g["y_true"].to_numpy()
    yp = g["pred"].to_numpy()

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())

    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", zero_division=0
    )

    rows.append({
        "test_scenario": sc,
        "seed": seed,
        "period": period,
        "n": len(g),
        "n_pos": int(yt.sum()),
        "predicted_positives": int(yp.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision_zero_division_0": p,
        "recall_zero_division_0": r,
        "f1_zero_division_0": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "f1_no_positive_convention": (
            1.0 if yt.sum() == 0 and yp.sum() == 0 else f1
        ),
    })

res = pd.DataFrame(rows)
res.to_csv(out, index=False)

print("Saved audit:", out)
print("\nClean-period summary:")
print(
    res[res.period == "clean"][
        ["n", "n_pos", "predicted_positives", "FP", "TN",
         "fpr", "f1_zero_division_0", "f1_no_positive_convention"]
    ].groupby(res[res.period == "clean"].test_scenario).mean().round(5)
)

print("\nClean-period rows with positive labels:")
print(
    res[(res.period == "clean") & (res.n_pos > 0)][
        ["test_scenario", "seed", "n_pos", "predicted_positives",
         "TP", "FP", "FN", "TN", "fpr", "f1_zero_division_0"]
    ].to_string(index=False)
)
