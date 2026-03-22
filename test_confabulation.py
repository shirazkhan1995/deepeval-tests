import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from datasets.loader import load_golden_data

GOLDEN_DATA = load_golden_data(category="confabulation")

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


@pytest.mark.parametrize(
    "record",
    GOLDEN_DATA,
    ids=[r["id"] for r in GOLDEN_DATA],
)
def test_confabulation(record):
    test_case = LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
        context=record.get("context"),
    )
    assert_test(test_case, [data_grounding, refusal_quality])
