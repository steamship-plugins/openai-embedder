"""Collection of object specifications used to communicate with the NLPCloud API."""

from steamship import SteamshipError

FAMILY_TO_DIMENSIONALITY = {
    "ada": 1024,
    "babbage": 2048,
    "curie": 4096,
    "davinci": 12288
}

MODEL_TO_DIMENSIONALITY = {
    "text-embedding-ada-002": 1536,
    **{
        f"text-similarity-{model}-001": dimensionality
        for model, dimensionality in FAMILY_TO_DIMENSIONALITY.items()
    },
    **{
        f"text-search-{model}-{type}-001": dimensionality
        for type in ["doc", "query"]
        for model, dimensionality in FAMILY_TO_DIMENSIONALITY.items()
    },
    **{
        f"code-search-{model}-{type}-001": FAMILY_TO_DIMENSIONALITY[model]
        for type in ["code", "text"]
        for model in ["babbage", "ada"]
    }
}


def validate_model(model: str):
    if model not in MODEL_TO_DIMENSIONALITY:
        raise SteamshipError(
            message=f"Model {model} is not supported by this plugin.. " +
                    f"Valid models for this task are: {[m for m in MODEL_TO_DIMENSIONALITY]}."
        )
