from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_token: str | None = None
    max_tokens: int = 300
    temperature: float = 0.2

settings = Settings()
