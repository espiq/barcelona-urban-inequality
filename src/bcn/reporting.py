"""Writers shared by both parts of the study: tables and headline numbers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_table(df: pd.DataFrame, name: str, folder: Path, floatfmt: str = ".3f") -> None:
    """Save ``df`` as CSV (for machines) and Markdown (for the report)."""
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    md = df.to_markdown(index=False, floatfmt=floatfmt)
    (folder / f"{name}.md").write_text(md, encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def save_results(results: dict, path: Path) -> None:
    """Write headline numbers as UTF-8 JSON."""
    payload = json.dumps(results, indent=2, default=_json_default, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
