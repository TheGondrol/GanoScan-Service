"""Response models. Field names are camelCase to match the Android app's JSON
contract exactly (the dicts we build already use these keys)."""

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Shape of every error response (FastAPI's default HTTPException body)."""

    detail: str


class Stage(BaseModel):
    index: str
    title: str
    sub: str
    done: bool = False
    image: str | None = None  # base64-encoded JPEG, no "data:" prefix; None if not produced


class ScanResult(BaseModel):
    """The full /predict response. Stateless — no id/timestamp/image-url
    fields, since the service persists nothing; the client owns its own
    history and generates whatever local identifiers it needs."""

    verdict: str
    label: str
    predictedClass: str
    severity: str
    confidence: float
    probabilities: dict[str, float]
    gamma: float
    inputResolution: str
    recommendations: list[str]
    stages: list[Stage]
    mock: bool


class ModelInfo(BaseModel):
    mode: str
    loaded: bool
    backbone: str
    classes: list[str]
    imageSize: list[int]
    device: str
    testAccuracy: float | None = None
    autoencoderLoaded: bool = False


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    service: str
    version: str
    model: ModelInfo
