import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from datasets.loader import load_golden_data

GOLDEN_DATA = load_golden_data()

correctness_metric = GEval(
    name="Correctness",
    criteria="Check correctness.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)


@pytest.mark.parametrize(
    "record",
    GOLDEN_DATA,
    ids=[r["id"] for r in GOLDEN_DATA],
)
def test_golden_dataset(record):
    test_case = LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
    )
    assert_test(test_case, [correctness_metric])
