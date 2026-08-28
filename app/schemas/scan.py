"""Response models. Field names are camelCase to match the Android app's JSON
contract exactly (the dicts we build/store already use these keys)."""

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """Shape of every error response (FastAPI's default HTTPException body)."""

    detail: str


class Stage(BaseModel):
    index: str
    title: str
    sub: str
    done: bool = False


class ScanResult(BaseModel):
    id: str
    treeId: str
    block: str
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
    imageUrl: str | None = None
    createdAt: str
    time: str
    dateLabel: str
    dayGroup: str
    mock: bool


class HistoryResponse(BaseModel):
    scans: list[ScanResult]
    count: int


class DeleteOneResult(BaseModel):
    deleted: bool
    id: str


class DeleteManyResult(BaseModel):
    deleted: int


class Stats(BaseModel):
    totalScan: int
    totalHealthy: int
    totalInitialInfection: int
    totalInfected: int


class ModelInfo(BaseModel):
    mode: str
    loaded: bool
    backbone: str
    classes: list[str]
    imageSize: list[int]
    device: str
    testAccuracy: float | None = None


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    service: str
    version: str
    model: ModelInfo
