"""Steamship OpenAI Embeddings Client"""

from typing import List, Optional, Type

from steamship import Tag
from steamship.invocable import Config, Invocable, create_handler
from steamship.plugin.request import PluginRequest

from openai.api_spec import validate_model
from openai.client import OpenAIEmbeddingClient
from tagger.span import Granularity, Span
from tagger.span_tagger import SpanStreamingConfig, SpanTagger


class OpenAIEmbedderConfig(Config):
    api_key: Optional[str]
    model: str = "text-similarity-curie-001"
    replace_newlines: bool = True

    granularity: Granularity = Granularity.BLOCK
    kind_filter: Optional[str] = None
    name_filter: Optional[str] = None

    class Config:
        use_enum_values = False


class OpenAIEmbedderPlugin(SpanTagger, Invocable):
    config: OpenAIEmbedderConfig
    client: OpenAIEmbeddingClient

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        validate_model(self.config.model)
        self.client = OpenAIEmbeddingClient(key=self.config.api_key)

    def config_cls(self) -> Type[OpenAIEmbedderConfig]:
        return OpenAIEmbedderConfig

    def get_span_streaming_args(self) -> SpanStreamingConfig:
        return SpanStreamingConfig(
            granularity=self.config.granularity,
            kind_filter=self.config.kind_filter,
            name_filter=self.config.name_filter
        )

    def tag_span(self, request: PluginRequest[Span]) -> List[Tag.CreateRequest]:
        if request.data.text.strip():
            tags_lists: List[List[Tag.CreateRequest]] = self.client.request(
                model=self.config.model,
                inputs=[request.data.text],
            )
            tags = tags_lists[0] or []
            return tags
        else:
            return []


handler = create_handler(OpenAIEmbedderPlugin)
