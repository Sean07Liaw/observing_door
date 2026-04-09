from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="observing-door", alias="APP_NAME")
    app_env: str = Field(default="dev", alias="APP_ENV")
    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")
    raw_image_dir: Path = Field(default=Path("./data/images/raw"), alias="RAW_IMAGE_DIR")
    processed_image_dir: Path = Field(default=Path("./data/images/processed"), alias="PROCESSED_IMAGE_DIR")
    clip_dir: Path = Field(default=Path("./data/clips"), alias="CLIP_DIR")
    timezone: str = Field(default="Asia/Taipei", alias="TIMEZONE")


settings = Settings()