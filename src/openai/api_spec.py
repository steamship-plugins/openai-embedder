"""Collection of object specifications used to communicate with the NLPCloud API."""

from steamship import SteamshipError

DIMENSIONS = {
    "ada": 1024,
    "babbage": 2048,
    "curie": 4096,
    "davinci": 12288,
}

MODEL_TO_FAMILY = {
    "text-similarity-ada-001": "ada",
    "text-similarity-babbage-001": "babbage",
    "text-similarity-curie-001": "curie",
    "text-similarity-davinci-001": "davinci",
    "text-search-ada-doc-001": "ada",
    "text-search-ada-query-001": "ada",
    "text-search-babbage-doc-001": "babbage",
    "text-search-babbage-query-001": "babbage",
    "text-search-curie-doc-001": "curie",
    "text-search-curie-query-001": "curie",
    "text-search-davinci-doc-001": "davinci",
    "text-search-davinci-query-001": "davinci",
    "code-search-ada-code-001": "ada",
    "code-search-ada-text-001": "ada",
    "code-search-babbage-code-001": "babbage",
    "code-search-babbage-text-001": "babbage",
}

def validate_model(model: str):
    # We know from docs.nlpcloud.com that only certain task<>model pairings are valid.
    if model not in MODEL_TO_FAMILY:
        raise SteamshipError(
            message=f"Model {model} is not supported by this plugin.. " +
            f"Valid models for this task are: {[m.value for m in MODEL_TO_FAMILY]}."
        )
