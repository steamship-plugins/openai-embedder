from abc import ABC, abstractmethod
from typing import List, Optional, Union

from steamship import Block, File, SteamshipError, Tag
from steamship.base.model import CamelModel
from steamship.invocable import InvocableResponse, post
from steamship.invocable.plugin_service import PluginService
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest

from tagger.span import Granularity, Span


class SpanStreamingConfig(CamelModel):
    granularity: Granularity
    kind_filter: Optional[str]
    name_filter: Optional[str]

class SpanTagger(PluginService[BlockAndTagPluginInput, BlockAndTagPluginOutput], ABC):
    """An implementation of a Tagger that permits implementors to care only about Spans."""

    def run(
        self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> Union[InvocableResponse[BlockAndTagPluginOutput], BlockAndTagPluginOutput]:
        args = self.get_span_streaming_args()

        spans = [
            span for span in Span.stream_from(
                file=request.data.file,
                granularity=args.granularity,
                kind_filter=args.kind_filter,
                name_filter=args.name_filter
            )
        ]
        output_tags = self.tag_spans(
            PluginRequest(
                data=spans,
                context=request.context,
                status=request.status,
                is_status_check=request.is_status_check
            )
        )

        # Now prepare the results. There's a bit of bookkeeping we have to do to make sure this is
        # structured properly with respect to the current BlockAndTag contract.
        block_lookup = {}
        output = BlockAndTagPluginOutput(file=File.CreateRequest(), tags=[])
        had_empty_block_ids = False
        for block in request.data.file.blocks:
            output_block = Block.CreateRequest(id=block.id, tags=[])
            if block.id is None:
                had_empty_block_ids = True
            else:
                block_lookup[block.id] = output_block
            output.file.blocks.append(output_block)

        # Go through each span and add to the appropriate place.
        for tag in output_tags:
            if request.data.file.id is not None and tag.file_id is None:
                raise SteamshipError(message="All Tags should have a file_id field")

            # Make sure the block_id has been provided correctly
            if args.granularity == Granularity.FILE:
                if tag.block_id is not None:
                    raise SteamshipError(message="A tag with a granularity of FILE should not have a block_id")
            else:
                if not had_empty_block_ids:
                    if tag.block_id is None:
                        raise SteamshipError(message="A tag with a granularity of BLOCK, BLOCK_TEXT, or TAG should have a block_id")
                    if tag.block_id not in block_lookup:
                        raise SteamshipError(message=f"The referenced block_id {tag.block_id} was not among the input Blocks")

            # Make sure the start_idx and end_idx have been provided correctly
            if args.granularity == Granularity.FILE:
                if tag.start_idx is not None:
                    raise SteamshipError(message="A Tag with a granularity of FILE or BLOCK should not have a start_idx")
                if tag.end_idx is not None:
                    raise SteamshipError(message="A Tag with a granularity of FILE or BLOCK should not have a end_idx")
            else:
                if tag.start_idx is None:
                    raise SteamshipError(message="A Tag with a granularity of BLOCK_TEXT or TAG should have a start_idx")
                if tag.end_idx is None:
                    raise SteamshipError(message="A Tag with a granularity of BLOCK_TEXT or TAG should have an end_idx")

            # Finally, add the tag.
            if args.granularity == Granularity.FILE:
                output.file.tags.append(tag)
            else:
                if tag.block_id is not None:
                    block_lookup[tag.block_id].tags.append(tag)
                else:
                    # This is technically a bug; but is a quick fix to work with our embedding index
                    output.file.blocks[0].tags.append(tag)

        # Finally, we can return the output
        return output


    @abstractmethod
    def get_span_streaming_args(self) -> SpanStreamingConfig:
        """This is a kludge to let the implementor return the required information for extracting Spans from the
        BlockAndTagPluginInput. Right now these have to be provided via the Config block on the plugin."""
        raise NotImplementedError()

    def tag_spans(self, request: PluginRequest[List[Span]]) -> List[Tag.CreateRequest]:
        all_tags = []
        for span in request.data:
            plugin_request = PluginRequest(
                data=span,
                context=request.context,
                status=request.status,
                is_status_check=request.is_status_check
            )
            tags = self.tag_span(plugin_request)
            for tag in tags:
                tag.file_id = span.file_id
                if span.granularity != Granularity.FILE:
                    tag.block_id = span.block_id
                if span.granularity == Granularity.BLOCK or span.granularity == Granularity.TAG:
                    tag.start_idx = span.start_idx
                    tag.end_idx = span.end_idx
                else:
                    tag.start_idx = None
                    tag.end_idx = None

                all_tags.append(tag)
        return all_tags

    @abstractmethod
    def tag_span(self, request: PluginRequest[Span]) -> List[Tag.CreateRequest]:
        """The plugin author now just has to implement tagging over the provided spans."""
        raise NotImplementedError()

    @post("tag")
    def run_endpoint(self, **kwargs) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Exposes the Tagger's `run` operation to the Steamship Engine via the expected HTTP path POST /tag"""
        return self.run(PluginRequest[BlockAndTagPluginInput].parse_obj(kwargs))
