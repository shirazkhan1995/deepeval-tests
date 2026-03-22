import math
from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.embedding_models.openai_embedding_model import OpenAIEmbeddingModel


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. No numpy dependency."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CosineSimilarityMetric(BaseMetric):
    """
    Embedding-based cosine similarity between actual_output and expected_output.
    Uses OpenAI's text-embedding-3-small by default; any DeepEvalBaseEmbeddingModel
    can be injected.
    """

    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]

    def __init__(
        self,
        threshold: float = 0.85,
        embedding_model: Optional[DeepEvalBaseEmbeddingModel] = None,
        model_name: str = "text-embedding-3-small",
        async_mode: bool = True,
    ):
        self.threshold = threshold
        self.async_mode = async_mode
        self.embedding_model = embedding_model or OpenAIEmbeddingModel(model=model_name)

    @property
    def __name__(self):
        return "Cosine Similarity"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        embeddings = self.embedding_model.embed_texts(
            [test_case.actual_output, test_case.expected_output]
        )
        self.score = _cosine_similarity(embeddings[0], embeddings[1])
        self.success = self.score >= self.threshold
        self.reason = f"Cosine similarity: {self.score:.4f} (threshold: {self.threshold})"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        embeddings = await self.embedding_model.a_embed_texts(
            [test_case.actual_output, test_case.expected_output]
        )
        self.score = _cosine_similarity(embeddings[0], embeddings[1])
        self.success = self.score >= self.threshold
        self.reason = f"Cosine similarity: {self.score:.4f} (threshold: {self.threshold})"
        return self.score

    def is_successful(self) -> bool:
        return self.success if self.success is not None else False
