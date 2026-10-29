import json
from pathlib import Path

nb_path = Path("analysis/full_capstone.ipynb")
out_path = Path("docs/model_pipeline_audit.md")

with nb_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

terms = [
    "gnn_edge_multi",
    "gnn_edge_multi_scratch",
    "EdgeNIDSClassifier",
    "GraphSAGEEncoder",
    "GraphMAE",
    "pretrained_encoder.pt",
    "load_state_dict",
    "torch.load",
    "torch.save",
    "train_model",
    "evaluate_logits",
    "finetune",
    "main_model",
    "main_model_finetuned.pt",
    "final_model_hpo_tuned.pt",
    "gated_fusion_model.pt",
    "StandardScaler",
    "fit_transform",
]

lines = ["# Model Pipeline Audit", ""]

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue

    src = "".join(cell.get("source", []))
    matched = [t for t in terms if t.lower() in src.lower()]

    if matched:
        lines += [
            f"## Cell {i}",
            f"**Matches:** {', '.join(matched)}",
            "",
            "```python",
            src[:6000],
            "```",
            ""
        ]

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Audit written: {out_path}")
print(f"Matching cells: {sum(1 for c in nb.get('cells', []) if c.get('cell_type') == 'code' and any(t.lower() in ''.join(c.get('source', [])).lower() for t in terms))}")
print(f"Report size: {out_path.stat().st_size:,} bytes")
