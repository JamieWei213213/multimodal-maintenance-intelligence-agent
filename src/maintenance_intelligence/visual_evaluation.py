from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return max(0.0, (centre - margin) / denominator), min(1.0, (centre + margin) / denominator)


def evaluate_visual_labels(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["asset_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Visual label set contains duplicate asset IDs.")
    invalid = [row["asset_id"] for row in rows if row["is_meaningful_visual"] not in {"yes", "no"}]
    if invalid:
        raise ValueError(f"Visual label set has incomplete labels: {invalid}")
    meaningful = sum(row["is_meaningful_visual"] == "yes" for row in rows)
    lower, upper = _wilson_interval(meaningful, len(rows))
    group_counts = Counter((row["source_filename"], row["asset_type"]) for row in rows)
    return {
        "sample_size": len(rows),
        "meaningful_candidates": meaningful,
        "sampled_candidate_precision": meaningful / len(rows) if rows else 0.0,
        "precision_wilson_95_lower": lower,
        "precision_wilson_95_upper": upper,
        "category_counts": dict(sorted(Counter(row["visual_category"] for row in rows).items())),
        "strata": [
            {"source_filename": source, "asset_type": asset_type, "count": count}
            for (source, asset_type), count in sorted(group_counts.items())
        ],
        "limitations": [
            "The deterministic sample contains only 24 candidates from one product family.",
            "This measures whether detected regions are meaningful visuals, not diagram comprehension or retrieval quality.",
            "Labels were assigned by visual inspection and have not been independently double-annotated.",
        ],
    }
