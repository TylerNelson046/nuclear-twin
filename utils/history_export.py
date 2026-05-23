"""Export orchestrator history dicts to CSV for UI download actions."""

from __future__ import annotations

import csv
import io
from typing import Any


def history_to_csv_text(history: dict[str, list[Any]]) -> str:
    """Serialize a time-aligned history mapping to CSV text.

    Rows are aligned by index across all series. The ``time_s`` column is placed
    first when present. Empty histories return a header-only file.
    """
    if not history:
        return ""

    keys = sorted(k for k, values in history.items() if isinstance(values, list))
    if "time_s" in keys:
        keys.remove("time_s")
        keys.insert(0, "time_s")

    if not keys:
        return ""

    n_rows = max((len(history[k]) for k in keys), default=0)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(keys)
    for idx in range(n_rows):
        writer.writerow([_cell(history[k], idx) for k in keys])
    return buffer.getvalue()


def _cell(series: list[Any], index: int) -> Any:
    if index < len(series):
        return series[index]
    return ""
