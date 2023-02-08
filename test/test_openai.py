import pytest
from steamship.data import TagValueKey

from openai.api_spec import MODEL_TO_DIMENSIONALITY

__copyright__ = "Steamship"
__license__ = "MIT"

from util import openai

from openai.client import OpenAIEmbeddingClient

TEST_DATA = []
for m in MODEL_TO_DIMENSIONALITY:
    TEST_DATA.append((m, MODEL_TO_DIMENSIONALITY[m]))


@pytest.mark.usefixtures("openai")
@pytest.mark.parametrize("model,dimensions", TEST_DATA)
def test_embed(openai: OpenAIEmbeddingClient, model: str, dimensions: int):
    texts = ["apple", "orange", "banana", "kiwi", "blueberry", "car"]
    res = openai.request(model, texts)
    assert len(res) == len(texts)
    for tags in res:
        assert len(tags) == 1
        tag = tags[0]
        assert tag.value is not None
        assert tag.value[TagValueKey.VECTOR_VALUE] is not None
        assert len(tag.value[TagValueKey.VECTOR_VALUE]) == dimensions
