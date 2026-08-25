from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import IngestionConfig, run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract provenance-rich chunks from locked maintenance manuals.")
    parser.add_argument("--manual-dir", type=Path, default=Path("data/manuals"))
    parser.add_argument("--checksums", type=Path, default=Path("evaluation/corpus_checksums.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--max-chunk-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=180)
    parser.add_argument("--render-pages", action="store_true")
    parser.add_argument("--render-dpi", type=int, default=144)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = IngestionConfig(
        min_text_chars=args.min_text_chars,
        ocr_dpi=args.ocr_dpi,
        max_chunk_chars=args.max_chunk_chars,
        overlap_chars=args.overlap_chars,
        render_pages=args.render_pages,
        render_dpi=args.render_dpi,
    )
    manifest = run_ingestion(args.manual_dir, args.checksums, args.output_dir, config=config)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()

