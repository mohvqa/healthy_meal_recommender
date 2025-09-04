from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
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

    class Config:
        env_file = ".env"

settings = Settings()
