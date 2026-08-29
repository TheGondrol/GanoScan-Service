from fastapi import APIRouter, Depends, File, HTTPException, Security, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import model_dep
from app.core.config import settings
from app.core.security import api_key_scheme
from app.schemas.scan import ErrorDetail, ScanResult
from app.services.inference import GanodermaModel
from app.services.verdict import build_result

router = APIRouter(tags=["inference"], dependencies=[Security(api_key_scheme)])


@router.post(
    "/predict",
    response_model=ScanResult,
    summary="Classify an uploaded photo",
    operation_id="predict",
    responses={
        400: {"model": ErrorDetail, "description": "Empty or unreadable image"},
        401: {"model": ErrorDetail, "description": "Missing or invalid X-API-Key"},
        413: {"model": ErrorDetail, "description": "Image exceeds the configured size limit"},
        422: {"description": "Missing/invalid form fields (FastAPI validation)"},
        500: {"model": ErrorDetail, "description": "Model inference failed"},
    },
)
async def predict(
    image: UploadFile = File(..., description="Leaf/stem photo (jpg, jpeg, png, bmp or webp)"),
    model: GanodermaModel = Depends(model_dep),
):
    """Classifies the uploaded image and returns the result. Stateless: nothing
    is persisted server-side — the client is responsible for its own history."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    if len(image_bytes) > settings.max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"image exceeds {settings.max_mb} MB limit")

    try:
        prob_map, predicted, confidence = await run_in_threadpool(model.predict, image_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    result = build_result(prob_map, predicted, confidence, image_bytes, model)
    result["mock"] = model.is_random
    return result
