import logging
from enum import Enum

from dotenv import find_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ConfigService(BaseSettings):
    """Centralized configuration service using Pydantic Settings."""

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=6543, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="password", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="beer_rag", alias="POSTGRES_DB")

    hf_home: str | None = Field(default=None, alias="HF_HOME")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    llm_provider: LLMProvider = Field(default=LLMProvider.GOOGLE, alias="LLM_PROVIDER")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="z-ai/glm-4.5-air:free",
        alias="OPENROUTER_MODEL",
    )

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def check_provider_keys(self) -> "ConfigService":
        if self.llm_provider == LLMProvider.GOOGLE and not (
            self.google_api_key and self.google_api_key.strip()
        ):
            raise ValueError("GOOGLE_API_KEY must be set when LLM_PROVIDER is 'google'")
        if self.llm_provider == LLMProvider.OPENROUTER and not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY must be set when LLM_PROVIDER is 'openrouter'"
            )
        return self

    @property
    def connection_string(self) -> str:
        """Constructs the SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
