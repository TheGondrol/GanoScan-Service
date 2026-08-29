"""Application settings, loaded from environment / .env (pydantic-settings).

All vars use the ``GANOSCAN_`` prefix, e.g. ``GANOSCAN_PORT=8000``.
Paths are resolved relative to the service root (this folder's grandparent).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .../GanoScan Service/app/core/config.py -> parents[2] == "GanoScan Service"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GANOSCAN_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),  # allow field names starting with "model_"
    )

    # --- Model artifacts (copied in from the training notebook's output) ---
    model_dir: str = "./models"
    jit_model_name: str = "model_jit.pt"
    state_dict_name: str = "best_model.pth"
    model_info_name: str = "model_info.json"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 5005

    # --- Inference / display ---
    # Fallback only — overridden by models/model_info.json when present.
    default_classes: list[str] = ["Healthy", "Infected", "Initial Infection"]
    default_image_size: tuple[int, int] = (160, 160)  # (height, width)
    default_backbone: str = "resnet50"
    imagenet_mean: list[float] = [0.485, 0.456, 0.406]
    imagenet_std: list[float] = [0.229, 0.224, 0.225]
    gamma: float = 0.8              # displayed gamma for the pipeline stage label
    max_mb: int = 15

    # --- API auth (all endpoints except /health and the docs) ---
    # If set, every request needs a matching `X-API-Key` header. Unset
    # (default) leaves the API open, which is fine for local dev but must be
    # set before exposing the service publicly.
    api_key: str | None = None

    def _abs(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path)

    @property
    def model_dir_path(self) -> Path:
        return self._abs(self.model_dir).resolve()


settings = Settings()
