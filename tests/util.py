import os
from tests import DOT_STEAMSHIP

import pytest
import toml
from steamship import SteamshipError

from openai.client import OpenAIEmbeddingClient


def read_test_file(filename: str):
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", filename), "r") as f:
        return f.read()


def get_key() -> str:
    try:
        secret_kwargs = toml.load(DOT_STEAMSHIP / "secrets.toml")
        return secret_kwargs.get("api_key")
    except Exception:
        print("Unable to get api_key from src/resources/api_key.json")


@pytest.fixture()
def openai() -> OpenAIEmbeddingClient:
    """Return an OpenAIEmbeddingClient.

    To use, simply import this file and then write a tests which takes `openai`
    as an argument.

    Example
    -------
    The client can be used by injecting a fixture as follows::

        @pytest.mark.usefixtures("openai")
        def test_something(openai):
          pass
    """

    environ_key = get_key()
    if environ_key is not None:
        return OpenAIEmbeddingClient(key=environ_key)
    raise SteamshipError(
        message="No api_key found. Please set the api_key variable in git ignored src/.steamship/secrets.toml"
    )
