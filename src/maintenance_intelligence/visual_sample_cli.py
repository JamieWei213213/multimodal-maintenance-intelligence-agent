from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.visual_sample import create_annotation_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible visual-candidate annotation sample.")
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/visual_assets.jsonl"))
    parser.add_argument("--manual-dir", type=Path, default=Path("data/manuals"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/visual_annotations.csv"))
    parser.add_argument("--render-dir", type=Path, default=Path("evaluation/visual_samples"))
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=580)
    args = parser.parse_args()
    result = create_annotation_sample(
        args.inventory,
        args.manual_dir,
        args.output,
        args.render_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
