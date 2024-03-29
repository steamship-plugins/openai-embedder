"""Steamship OpenAI Embeddings Client"""
from typing import List, Optional, Type, Dict, Any

from pydantic import Field
from steamship import Tag, Steamship, SteamshipError
from steamship.invocable import Config, Invocable, InvocationContext
from steamship.plugin.outputs.plugin_output import UsageReport
from steamship.plugin.request import PluginRequest

from openai.api_spec import validate_model
from openai.client import OpenAIEmbeddingClient
from tagger.span import Granularity, Span
from tagger.span_tagger import SpanStreamingConfig, SpanTagger

VALID_MODELS_FOR_BILLING = ["text-embedding-ada-002"]

class OpenAIEmbedderPlugin(SpanTagger, Invocable):

    class OpenAIEmbedderConfig(Config):
        api_key: Optional[str] = Field("", description="Description")
        model: str = Field("text-embedding-ada-002", description="Description")
        replace_newlines: bool = Field(True, description="Replace newlines with spaces")
        granularity: Granularity = Field(Granularity.BLOCK.value, description="Granularity level")
        kind_filter: Optional[str] = Field("", description="Filter tags on kind")
        name_filter: Optional[str] = Field("", description="Filter tags on name")
        dimensionality: int = Field(None, description="Dimensionality of the embeddings")

        class Config:
            use_enum_values = False

    config: OpenAIEmbedderConfig
    client: OpenAIEmbeddingClient

    def __init__(self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
     ):
        # Load original api key before it is read from TOML, so we know to restrict models for billing
        original_api_key = config['api_key']
        super().__init__(client, config, context)
        if original_api_key == "" and self.config.model not in VALID_MODELS_FOR_BILLING:
            raise SteamshipError(f"This plugin cannot be used with model {self.config.model} while using Steamship's API key. Valid models are {VALID_MODELS_FOR_BILLING}")
        validate_model(self.config.model)
        self.client = OpenAIEmbeddingClient(key=self.config.api_key)

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.OpenAIEmbedderConfig

    def get_span_streaming_args(self) -> SpanStreamingConfig:
        return SpanStreamingConfig(
            granularity=self.config.granularity,
            kind_filter=self.config.kind_filter,
            name_filter=self.config.name_filter
        )

    def tag_span(self, request: PluginRequest[Span]) -> (List[Tag], Optional[List[UsageReport]]):
        if request.data.text.strip():
            tags_lists, usage = self.client.request(
                model=self.config.model,
                inputs=[request.data.text],
            )
            tags = tags_lists[0] or []
            return tags, usage
        else:
            return [], None
