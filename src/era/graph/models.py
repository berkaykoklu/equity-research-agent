from langchain_core.language_models import BaseChatModel

DRAFTING_MODEL = "gpt-5.6-terra"
CHEAP_MODEL = "gpt-5.6-luna"

# Same reasoning as EdgarClient's 30s httpx timeout: an ungated network call
# in a batch run (twenty filings, each a boundary-selection call) can stall
# indefinitely on a dropped connection with no timeout set at all. A couple
# of quiet retries absorb the kind of transient blip SEC's own client
# already tolerates; anything past that surfaces rather than hanging.
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 2


def build_model(
    provider: str = "openai",
    model_name: str = DRAFTING_MODEL,
    temperature: float = 0.0,
) -> BaseChatModel:
    """One seam for the provider. Swapping is a config change, not a rewrite."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
    if provider == "anthropic":
        # Not a project dependency today -- nothing in this codebase uses
        # the Anthropic provider yet, so the package isn't installed and
        # mypy can't resolve it. The import stays local to this branch so
        # importing this module never requires langchain_anthropic to exist.
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]

        return ChatAnthropic(  # type: ignore[no-any-return]
            model=model_name,
            temperature=temperature,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
        )
    raise ValueError(f"unknown provider: {provider}")
