import os
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.deps import model_dep
from app.core.config import settings
from app.core.security import api_key_scheme
from app.db import store
from app.schemas.scan import ErrorDetail, ScanResult
from app.services.inference import GanodermaModel
from app.services.verdict import build_result
from app.utils.scan_meta import date_label, day_group, new_id

router = APIRouter(tags=["inference"], dependencies=[Security(api_key_scheme)])

_ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _write_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


@router.post(
    "/predict",
    response_model=ScanResult,
    status_code=201,
    summary="Classify an uploaded photo and persist the scan",
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
    treeId: str | None = Form(None, description="e.g. 'Pohon #A-142'; defaults to 'Pohon #<generated id>'"),
    block: str | None = Form(None, description="e.g. 'Blok C'; defaults to GANOSCAN_DEFAULT_BLOCK"),
    model: GanodermaModel = Depends(model_dep),
):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image")
    if len(image_bytes) > settings.max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"image exceeds {settings.max_mb} MB limit")

    try:
        prob_map, predicted, confidence = await run_in_threadpool(model.predict, image_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    scan_id = new_id()
    now = datetime.now()

    # Persist the original image for the before/after view.
    image_url = None
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in _ALLOWED_EXT:
        ext = ".jpg"
    fname = f"{scan_id}{ext}"
    try:
        await run_in_threadpool(_write_file, str(settings.upload_dir / fname), image_bytes)
        image_url = f"/uploads/{fname}"
    except Exception as e:  # noqa: BLE001
        print(f"[predict] could not save image: {e}")

    result = build_result(prob_map, predicted, confidence, image_bytes, model)
    scan = {
        "id": scan_id,
        "treeId": treeId or f"Pohon #{scan_id}",
        "block": block or settings.default_block,
        "createdAt": now.isoformat(timespec="seconds"),
        "time": now.strftime("%H:%M"),
        "dateLabel": date_label(now),
        "dayGroup": day_group(now),
        "imageUrl": image_url,
        "mock": model.is_random,
        **result,
    }
    await run_in_threadpool(store.insert_scan, scan)
    return scan
