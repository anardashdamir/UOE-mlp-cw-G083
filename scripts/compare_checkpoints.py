"""Print EM scores for all checkpoints across all model/thinking combinations."""

import json
from pathlib import Path

output_dir = Path("output")

for model_dir in sorted(output_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    for thinking_dir in sorted(model_dir.iterdir()):
        if not thinking_dir.is_dir():
            continue
        metrics_dir = thinking_dir / "metrics"
        if not metrics_dir.exists():
            continue

        print(f"\n{'='*60}")
        print(f"  {model_dir.name} / {thinking_dir.name}")
        print(f"{'='*60}")
        print(f"  {'Checkpoint':<25} {'EM':>7} {'F1':>7} {'FA':>7} {'SV':>7}")
        print(f"  {'-'*53}")

        results = []
        for f in sorted(metrics_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            o = data["overall"]
            name = f.stem.replace("eval_", "").replace("_fp16", "")
            results.append((name, o.get("exact_match", 0), o.get("f1", 0), o.get("field_accuracy", 0), o.get("structural_validity", 0)))

        best_em = max(r[1] for r in results) if results else 0
        for name, em, f1, fa, sv in results:
            marker = " <-- BEST" if em == best_em else ""
            print(f"  {name:<25} {em:>7.3f} {f1:>7.3f} {fa:>7.3f} {sv:>7.3f}{marker}")
