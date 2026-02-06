from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigService(BaseSettings):
    """Centralized configuration service using Pydantic Settings."""

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=6543, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="password", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="beer_rag", alias="POSTGRES_DB")

    hf_home: str | None = Field(default=None, alias="HF_HOME")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    @property
    def connection_string(self) -> str:
        """Constructs the SQLAlchemy connection string."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
