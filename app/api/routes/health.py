from fastapi import APIRouter, Depends

from app.api.deps import model_dep
from app.schemas.scan import Health
from app.services.inference import GanodermaModel

router = APIRouter(tags=["meta"])


@router.get(
    "/health",
    response_model=Health,
    summary="Service + model status",
    description="Always reachable without an API key, for load balancer / uptime checks.",
    operation_id="getHealth",
)
async def health(model: GanodermaModel = Depends(model_dep)):
    return {
        "status": "ok",
        "service": "ganoscan-service",
        "version": "1.0.0",
        "model": model.info(),
    }
