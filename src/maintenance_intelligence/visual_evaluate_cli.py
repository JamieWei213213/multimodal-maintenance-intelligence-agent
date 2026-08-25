from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.visual_evaluation import evaluate_visual_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the manually labeled visual candidate sample.")
    parser.add_argument("--labels", type=Path, default=Path("evaluation/visual_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/visual_metrics.json"))
    args = parser.parse_args()
    report = evaluate_visual_labels(args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
