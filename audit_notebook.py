import json
from pathlib import Path

notebook_path = Path("analysis/full_capstone.ipynb")
output_path = Path("docs/notebook_audit.md")

with notebook_path.open("r", encoding="utf-8") as f:
    notebook = json.load(f)

lines = [
    "# Notebook Audit",
    "",
    f"- Notebook: `{notebook_path}`",
    f"- Total cells: {len(notebook.get('cells', []))}",
    "",
    "## Code cell inventory",
    ""
]

terms = [
    "gnn_edge_multi",
    "GraphSAGEEncoder",
    "GraphMAE",
    "EdgeNIDSClassifier",
    "train_model",
    "evaluate_logits",
    "pretrain",
    "finetune",
    "Optuna",
    "Wilcoxon",
    "holm",
    "StandardScaler",
    "fit_transform",
]

for i, cell in enumerate(notebook.get("cells", [])):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") != "code":
        continue

    matched = [term for term in terms if term.lower() in source.lower()]
    if matched:
        lines.extend([
            f"### Cell {i}",
            f"- Matching terms: {', '.join(matched)}",
            "",
            "```python",
            source[:2500],
            "```",
            ""
        ])

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Notebook cells: {len(notebook.get('cells', []))}")
print(f"Audit saved to: {output_path}")
print(f"Audit size: {output_path.stat().st_size:,} bytes")
