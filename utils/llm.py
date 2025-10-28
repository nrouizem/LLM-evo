from __future__ import annotations

from typing import Tuple

from openai import OpenAI

from utils.client import client as default_client

DEFAULT_MODEL = "gpt-5-mini"


def resolve_model_client(
        model: str | None = None,
        client: OpenAI | None = None
    ) -> Tuple[str, OpenAI]:
    """
    Return a `(model, client)` pair, defaulting to the shared client and the
    repo's preferred base model when overrides are not supplied.
    """
    resolved_client = client or default_client
    resolved_model = model or DEFAULT_MODEL
    return resolved_model, resolved_client
