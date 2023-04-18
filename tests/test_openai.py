import pytest
from steamship.data import TagValueKey
from steamship.plugin.outputs.plugin_output import OperationUnit

from openai.api_spec import MODEL_TO_DIMENSIONALITY
from openai.client import OpenAIEmbeddingClient

TEST_DATA = []
for m in MODEL_TO_DIMENSIONALITY:
    TEST_DATA.append((m, MODEL_TO_DIMENSIONALITY[m]))

from .util import openai

@pytest.mark.usefixtures("openai")
@pytest.mark.parametrize("model,dimensions", TEST_DATA)
def test_embed(openai: OpenAIEmbeddingClient, model: str, dimensions: int):
    texts = ["apple", "orange", "banana", "kiwi", "blueberry", "car"]
    res, _ = openai.request(model, texts)
    assert len(res) == len(texts)
    for tags in res:
        assert len(tags) == 1
        tag = tags[0]
        assert tag.value is not None
        assert tag.value[TagValueKey.VECTOR_VALUE] is not None
        assert len(tag.value[TagValueKey.VECTOR_VALUE]) == dimensions


@pytest.mark.usefixtures("openai")
def test_embed(openai: OpenAIEmbeddingClient):
    texts = ["apple", "orange sporange", "banana banana banana"]
    res, usages = openai.request("text-embedding-ada-002", texts)
    assert len(res) == len(texts)
    assert sum([ usage.operation_amount for usage in usages]) == 7
    assert len(usages) == 1
    assert usages[0].operation_unit == OperationUnit.PROMPT_TOKENS



