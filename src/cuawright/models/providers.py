from __future__ import annotations


PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-5.5",
    "openrouter-anthropic": "anthropic/claude-opus-5",
    "anthropic": "claude-opus-5",
}
PROVIDER_DEFAULT_BASE_URLS = {
    "openai": "https://maif-gateway.azure-api.net/api",
    "openrouter-anthropic": "https://openrouter.ai/api",
    "anthropic": "https://api.anthropic.com",
}


def resolve_model(provider: str, model: str | None) -> str:
    if model:
        return model
    try:
        return PROVIDER_DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported model provider: {provider!r}") from exc


def resolve_base_url(provider: str, base_url: str | None) -> str:
    if base_url:
        return base_url
    try:
        return PROVIDER_DEFAULT_BASE_URLS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported model provider: {provider!r}") from exc
