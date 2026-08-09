from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from the environment and .env.

    Instantiated with no arguments by the CLI. Only `edgar_user_agent` is
    required — every other field defaults to empty so a missing key only
    fails when something actually needs it.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    edgar_user_agent: str
    openai_api_key: str = ""
    voyage_api_key: str = ""
    database_url: str = ""
    opik_api_key: str = ""
    opik_workspace: str = ""
