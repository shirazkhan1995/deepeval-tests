import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from datasets.loader import load_golden_data

GOLDEN_DATA = load_golden_data()

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


@pytest.mark.parametrize(
    "record",
    GOLDEN_DATA,
    ids=[r["id"] for r in GOLDEN_DATA],
)
def test_llm_judge(record):
    test_case = LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
    )
    assert_test(test_case, [factual_grounding, relevance, completeness])
