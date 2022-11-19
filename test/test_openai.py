import pytest
from steamship.data import TagValue

from openai.api_spec import DIMENSIONS, MODEL_TO_FAMILY

__copyright__ = "Steamship"
__license__ = "MIT"

from util import openai

from openai.client import OpenAIEmbeddingClient

TEST_DATA = []
for m in MODEL_TO_FAMILY:
    family = MODEL_TO_FAMILY[m]
    TEST_DATA.append((m, DIMENSIONS[family]))


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
        assert tag.value[TagValue.VECTOR_VALUE] is not None
        assert len(tag.value[TagValue.VECTOR_VALUE]) == dimensions
