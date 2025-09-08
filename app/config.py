from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    data_dir: Path = Path("app/data")

    # Model Hyperparameters
    embedding_dim: int = 32
    emb_dropout: float = 0.2
    nhead: int = 2
    ff_dim: int = 32
    dropout_prob: float = 0.4
    n_layers: int = 2

    # Device
    device: str = "cpu"

    # Environment extras (used by start scripts)
    workers: int = 4
    log_level: str = "info"
    model_dir: Path = Path("./models")

settings = Settings()