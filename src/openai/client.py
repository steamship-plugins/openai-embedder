from enum import Enum
from typing import List

from pydantic import BaseModel
from steamship.data import TagKind, TagValue
from steamship.data.tags import Tag

from openai.api_spec import validate_model
from openai.request_utils import concurrent_json_posts


class OpenAIObject(str, Enum):
    LIST = 'list'
    EMBEDDING = 'embedding'

class OpenAIEmbedding(BaseModel):
    object: OpenAIObject # 'embedding'
    index: int
    embedding: List[float]

    def to_tag(self, model: str) -> Tag.CreateRequest:
        return Tag.CreateRequest(
            kind=TagKind.EMBEDDING,
            name=model,
            value={
                "service": "openai",
                TagValue.VECTOR_VALUE: self.embedding
            },
        )

class OpenAIEmbeddingList(BaseModel):
    object: OpenAIObject # 'list'
    data: List[OpenAIEmbedding]

    def to_tags(self, model: str) -> List[Tag.CreateRequest]:
        return [embedding.to_tag(model) for embedding in self.data]

class OpenAIEmbeddingClient:
    URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, key: str):
        self.key = key

    def request(
        self, model: str, inputs: List[str], **kwargs
    ) -> List[List[Tag.CreateRequest]]:
        """Performs an OpenAI request. Throw a SteamshipError in the event of error or empty response."""

        validate_model(model)

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        def items_to_body(items: List[str]):
            return {
                "model": model,
                "input": items
            }

        responses = concurrent_json_posts(self.URL, headers, inputs, 6, items_to_body, "openai")

        ret: List[List[Tag.CreateRequest]] = []
        for response in responses:
            obj = OpenAIEmbeddingList.parse_obj(response)
            for embedding in obj.data:
                ret.append([embedding.to_tag(model=model)])

        return ret
