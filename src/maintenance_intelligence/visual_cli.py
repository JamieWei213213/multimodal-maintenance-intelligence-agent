from __future__ import annotations

import argparse
import json
from pathlib import Path

from maintenance_intelligence.visuals import build_visual_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory citable visual assets in the manual corpus.")
    parser.add_argument("--manual-dir", type=Path, default=Path("data/manuals"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/visual_assets.jsonl"))
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_visual_inventory(args.manual_dir, args.output, render_dir=args.render_dir), indent=2))


if __name__ == "__main__":
    main()
