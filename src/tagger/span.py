"""Span class for streaming units of work to a Tagger for processing.

This class is being implemented in the NLPCloud Tagger to act as a proposal for broader inclusion in either the
Python Client or even the Engine as a native operation.

Sometimes a user wants to classify / embed / tag only a subset of a document. For example:
  - Embed each span of text tagged SENTENCE
  - Sentiment tag each span of text containing an ENTITY

Currently there is no standardized way to express this wish: it's up to the Tagger Plugin author to decide
how to handle it. This feels like something universal enough to begin playing an abstraction that could be
adopted globally. The Span class is a first draft sketch of doing that.

Here's how it works:

Given:

- a `File` object, as presented to the tagger, along with
- a desired unit of `Granularity` (FILE, BLOCK, TAG), and
- optional `kind` and `name` filters upon `Tag`

The `Span.stream_from` method generates a list of provenance-stamped text inputs that can be provided to
whatever operation the tagger implements.

For now, the `granularity`, `kind`, and `name` arguments can be provided in the `Config` of the `NLPTagger` class.
But the eventual location of such generation could be:

- Global to all Tagging plugins
- A helper function in `steamship.util`
- In the Engine itself

"""

from enum import Enum
from typing import Generator, List, Optional

from steamship import Block, File, Tag
from steamship.base.model import CamelModel


class Granularity(str, Enum):
    """
    FILE means "Tag with no block_id, no start_idx, and no end_idx"
    BLOCK means "Tags that all have start_idx=0 and end_idx=len(block.text)"
    TAG means "Tag with block_id, start_idx, and end_idx"
    """
    FILE = "file"
    BLOCK = "blocktext"
    TAG = "tag"

def _no_filter(kind_filter: str = None, name_filter: str = None) -> bool:
    return kind_filter is None and name_filter is None

def _tag_matches(tag: Tag, kind_filter: str = None, name_filter: str = None) -> Optional[Tag]:
    """Returns whether the tag matches the provided filter."""
    if (
        (kind_filter is None or tag.kind == kind_filter) and
        (name_filter is None or tag.name == name_filter)
    ):
        return tag
    return None

def _tags_match(tags: Optional[List[Tag]], kind_filter: str = None, name_filter: str = None) -> Optional[List[Tag]]:
    """Returns whether one of the tags matches the provided filter."""
    if tags is None:
        return None
    ret_tags = []
    for tag in tags:
        if tag := _tag_matches(tag, kind_filter=kind_filter, name_filter=name_filter):
            ret_tags.append(tag)

    if len(ret_tags) > 0:
        return ret_tags
    return None


def _file_matches(file: File, kind_filter: str = None, name_filter: str = None) -> Optional[List[Tag]]:
    """Returns whether one of the file tags matches the provided filter."""
    if not kind_filter and not name_filter:
        return None
    return _tags_match(file.tags, kind_filter=kind_filter, name_filter=name_filter)


def _block_matches(block: Block, kind_filter: str = None, name_filter: str = None) -> Optional[List[Tag]]:
    """Returns whether one of the block tags matches the provided filter."""
    if not kind_filter and not name_filter:
        return None
    return _tags_match(block.tags, kind_filter=kind_filter, name_filter=name_filter)


class Span(CamelModel):
    """A span covering a region of text that is to be processed by a Tagger.

    Attributes
    ----------
    file_id: str
        ID of the file from which this Span comes.
    block_id: Optional[str]
        ID of the block from which this Span comes. None if the Span covers every block in the file and therefore
        is intended to represents the File itself (e.g. classify this file) rather than the text within the file
        (classify this region of text).
    granularity: Granularity
        What level of granularity this span is intended to represent:

        - the entire File, producing file tags
        - an entire Block within the file, producing block tags with no start_idx or end_idx
        - a span of text within the file, producing block tags with start_idx or end_idx
    text: str
        The text covered by this span.
    start_idx: Optional[int]
        The start index of the span text.
        - For granularity FILE, this is None
        - For granularity BLOCK, this is 0
        - For granularity TAG, this is the start_idx of the span of text within its block
    end_idx: Optional[int]
        The end index of the span text.
        - For granularity FILE, this is None
        - For granularity BLOCK, this is len(block.text)
        - For granularity TAG, this is the end_idx of the span of text within its block
    related_tags: Optional[List[Tag]]
        The list of tags that caused this Span to be provided for consideration. For example, if this Span
        was presented for consideration because of the intersection of a SENTENCE with "Some Person's Name"
        and a SENTIMENT overlapping, then all three of those tags will be provided in case the Tagger chooses
        to incorporate them in the tagging.
    """
    file_id: Optional[str]
    block_id: Optional[str]
    granularity: Granularity
    text: str
    start_idx: Optional[int]
    end_idx: Optional[int]
    related_tags: Optional[List[Tag]]

    @staticmethod
    def stream_from(
            file: File = None,
            granularity: Granularity = None,
            kind_filter: str = None,
            name_filter: str = None
    ) -> Generator["Span", None, None]:
        """Steams units of work to be provided as input to the tagger.

        Only a simple mechanism for specifying the `related_tags` is provided at present.

        Attributes
        ----------
        file : File
            The provided file, from the Steamship Engine, from which units of processing come.
        granularity : Granularity
            The desired granularity of the units of work: the entire file, blocks, or text covered by tags
        kind_filter : str
            Whether to filter the unit of granularity for those matching coverage by Tags of that kind
        name_filter : str
            Whether to filter the unit of granularity for those matching coverage by Tags of that name
        """
        if not file:
            return

        if granularity == Granularity.FILE:
            tags = _file_matches(file, kind_filter=kind_filter, name_filter=name_filter)
            if tags or _no_filter(kind_filter, name_filter):
                all_text = "\n".join([block.text for block in file.blocks or [] if block.text])
                yield Span(
                    file_id = file.id,
                    granularity = Granularity.FILE,
                    text = all_text,
                    related_tags = tags or []
                )
        elif granularity == Granularity.TAG:
            if not file.blocks:
                return
            for block in file.blocks:
                if not block.tags:
                    continue
                for tag in block.tags:
                    matched_tag = _tag_matches(tag, kind_filter=kind_filter, name_filter=name_filter)
                    if matched_tag or _no_filter(kind_filter, name_filter):
                        yield Span(
                            file_id=file.id,
                            block_id=block.id,
                            granularity=Granularity.TAG,
                            text=block.text[tag.start_idx:tag.end_idx],
                            start_idx=tag.start_idx,
                            end_idx=tag.end_idx,
                            related_tags=[matched_tag] if matched_tag is not None else []
                        )
        elif granularity == Granularity.BLOCK:
            if not file.blocks:
                return
            for block in file.blocks:
                tags = _block_matches(block, kind_filter=kind_filter, name_filter=name_filter)
                if tags or _no_filter(kind_filter, name_filter):
                    yield Span(
                        file_id=file.id,
                        block_id=block.id,
                        granularity=Granularity.BLOCK,
                        text=block.text,
                        start_idx=0,
                        end_idx=len(block.text),
                        related_tags=tags or []
                    )
