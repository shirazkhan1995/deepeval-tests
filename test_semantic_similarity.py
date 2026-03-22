import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from datasets.loader import load_golden_data
from metrics.cosine_similarity import CosineSimilarityMetric

GOLDEN_DATA = load_golden_data()

similarity_metric = CosineSimilarityMetric(threshold=0.7)


@pytest.mark.parametrize(
    "record",
    GOLDEN_DATA,
    ids=[r["id"] for r in GOLDEN_DATA],
)
def test_semantic_similarity(record):
    test_case = LLMTestCase(
        input=record["input"],
        actual_output=record["actual_output"],
        expected_output=record["expected_output"],
    )
    assert_test(test_case, [similarity_metric])
