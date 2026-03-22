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


def _build_confabulation_metrics() -> list:
    data_grounding = GEval(
        name="Data Grounding",
        criteria="Evaluate whether the actual output stays grounded in the provided context and refuses to fabricate information about data that does not exist.",
        evaluation_steps=[
            "Check if the input asks about data, columns, tables, or metrics that do NOT exist according to the context.",
            "If the context indicates the requested data does not exist, verify the actual output acknowledges this limitation rather than fabricating an answer.",
            "Penalize any specific numbers, values, or data points that are not supported by the context.",
            "Reward responses that clearly state what IS available in the data model when declining to answer.",
            "Score 10 if the response correctly refuses and explains why, 5 if it partially refuses but includes some fabrication, 0 if it fabricates a full answer.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.7,
    )

    refusal_quality = GEval(
        name="Refusal Quality",
        criteria="Evaluate the quality and helpfulness of the model's refusal when asked about unavailable data.",
        evaluation_steps=[
            "Check if the response provides a clear, direct statement that the requested information is unavailable.",
            "Verify the response explains WHY the data is unavailable (e.g., column doesn't exist, out of scope).",
            "Check if the response helpfully suggests what IS available or what alternatives exist.",
            "Penalize vague refusals like 'I don't know' without explanation. Reward specific, actionable refusals.",
            "Score 10 if refusal is clear, explains the gap, and suggests alternatives. Score 5 if refusal is clear but unhelpful. Score 0 if no refusal or completely fabricated.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=0.7,
    )

    return [data_grounding, refusal_quality]


def _extract_scores(result) -> dict[str, dict[str, float | None]]:
    """Extract per-test-case per-metric scores from an EvaluationResult."""
    scores: dict[str, dict[str, float | None]] = {}
    for test_result in result.test_results:
        test_id = test_result.name
        scores[test_id] = {}
        for md in test_result.metrics_data or []:
            scores[test_id][md.name] = md.score
    return scores


def run_evaluation() -> dict[str, dict[str, float | None]]:
    """
    Run the full golden dataset through appropriate metrics per category.
    Standard records get the 4 original metrics; confabulation records get
    Data Grounding + Refusal Quality.
    Returns: {test_id: {metric_name: score}}
    """
    all_records = load_golden_data()
    standard_records = [r for r in all_records if r.get("category") != "confabulation"]
    confab_records = [r for r in all_records if r.get("category") == "confabulation"]

    scores: dict[str, dict[str, float | None]] = {}

    if standard_records:
        standard_cases = [
            LLMTestCase(
                name=r["id"],
                input=r["input"],
                actual_output=r["actual_output"],
                expected_output=r["expected_output"],
            )
            for r in standard_records
        ]
        result = deepeval.evaluate(
            test_cases=standard_cases,
            metrics=_build_metrics(),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
        scores.update(_extract_scores(result))

    if confab_records:
        confab_cases = [
            LLMTestCase(
                name=r["id"],
                input=r["input"],
                actual_output=r["actual_output"],
                expected_output=r["expected_output"],
                context=r.get("context"),
            )
            for r in confab_records
        ]
        result = deepeval.evaluate(
            test_cases=confab_cases,
            metrics=_build_confabulation_metrics(),
            display_config=DisplayConfig(print_results=False, show_indicator=False),
            cache_config=CacheConfig(write_cache=False, use_cache=False),
        )
        scores.update(_extract_scores(result))

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
