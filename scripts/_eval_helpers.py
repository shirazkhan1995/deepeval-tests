"""
Shared evaluation helpers for regression_check.py and update_baseline.py.
Runs the full golden dataset through all 4 metrics and returns scores keyed
by test case ID.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables the same way conftest.py does
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env.development", override=True)

import deepeval
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Make sure datasets/ and metrics/ are importable when running scripts directly
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.loader import load_golden_data
from metrics.cosine_similarity import CosineSimilarityMetric

BASELINE_PATH = _ROOT / "baselines" / "scores.json"


def _build_metrics() -> list:
    factual_grounding = GEval(
        name="Factual Grounding",
        criteria="Evaluate whether the actual output is factually correct compared to the expected output.",
        evaluation_steps=[
            "Compare each factual claim in the actual output against the expected output.",
            "Check for fabricated facts, incorrect numbers, wrong DAX functions, or misleading statements.",
            "Verify Power BI terminology (measures, columns, relationships, DAX functions) is used correctly.",
            "Score 10 if all facts match, 5 if partially correct, 0 if fundamentally wrong.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
    )

    relevance = GEval(
        name="Relevance",
        criteria="Evaluate whether the actual output directly answers the question asked in the input.",
        evaluation_steps=[
            "Read the input question and identify what is being asked.",
            "Check if the actual output addresses the specific question.",
            "Penalize responses that are correct but answer a different question.",
            "Score 10 if directly answers the question, 5 if partially relevant, 0 if off-topic.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )

    completeness = GEval(
        name="Completeness",
        criteria="Evaluate whether the actual output covers all aspects of the expected output.",
        evaluation_steps=[
            "Identify all key points in the expected output.",
            "Check which key points are present in the actual output.",
            "For Power BI answers, verify DAX formulas, instructions, or configuration details are complete.",
            "Score 10 if all aspects covered, 5 if major points present but details missing, 0 if most content missing.",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7,
    )

    cosine = CosineSimilarityMetric()

    return [factual_grounding, relevance, completeness, cosine]


def run_evaluation() -> dict[str, dict[str, float | None]]:
    """
    Run the full golden dataset through all metrics.
    Returns: {test_id: {metric_name: score}}
    """
    records = load_golden_data()
    metrics = _build_metrics()

    test_cases = [
        LLMTestCase(
            name=record["id"],
            input=record["input"],
            actual_output=record["actual_output"],
            expected_output=record["expected_output"],
        )
        for record in records
    ]

    result = deepeval.evaluate(
        test_cases=test_cases,
        metrics=metrics,
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )

    scores: dict[str, dict[str, float | None]] = {}
    for test_result in result.test_results:
        test_id = test_result.name
        scores[test_id] = {}
        for md in test_result.metrics_data or []:
            scores[test_id][md.name] = md.score

    return scores


def load_baseline() -> dict:
    """Load baselines/scores.json. Raises FileNotFoundError if missing."""
    with open(BASELINE_PATH) as f:
        return json.load(f)


def write_baseline(scores: dict[str, dict[str, float | None]]) -> None:
    """Write scores to baselines/scores.json, creating the directory if needed."""
    from datetime import datetime, timezone
    import importlib.metadata

    try:
        deepeval_version = importlib.metadata.version("deepeval")
    except Exception:
        deepeval_version = "unknown"

    all_metric_names: list[str] = []
    for metric_scores in scores.values():
        for name in metric_scores:
            if name not in all_metric_names:
                all_metric_names.append(name)

    baseline = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deepeval_version": deepeval_version,
            "test_case_count": len(scores),
            "metrics": all_metric_names,
        },
        "scores": scores,
    }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
