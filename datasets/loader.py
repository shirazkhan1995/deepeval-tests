"""
Golden dataset loader utilities for deepeval test suites.

Supports JSONL format (recommended for production) with optional filtering
by category, id prefix, or arbitrary field values.
"""

import json
from pathlib import Path
from typing import Optional

DATASETS_DIR = Path(__file__).parent


def load_golden_data(
    filename: str = "golden_data.jsonl",
    category: Optional[str] = None,
) -> list[dict]:
    """
    Load golden test cases from a JSONL file.

    Args:
        filename: Name of the JSONL file inside the datasets/ directory.
        category: If provided, only return records matching this category.

    Returns:
        List of record dicts with at minimum: id, input, actual_output, expected_output.
    """
    path = DATASETS_DIR / filename
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if category is None or record.get("category") == category:
                records.append(record)
    return records
