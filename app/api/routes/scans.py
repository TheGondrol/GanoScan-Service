import os

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.concurrency import run_in_threadpool

from app.api.deps import require_admin
from app.core.config import settings
from app.core.security import api_key_scheme
from app.db import store
from app.schemas.scan import (
    DeleteManyResult,
    DeleteOneResult,
    ErrorDetail,
    HistoryResponse,
    ScanResult,
    Stats,
)

router = APIRouter(tags=["scans"], dependencies=[Security(api_key_scheme)])


def _remove_upload(image_url: str | None):
    """Best-effort delete of the original photo backing a scan."""
    if not image_url:
        return
    path = settings.upload_dir / os.path.basename(image_url)
    try:
        if path.exists():
            path.unlink()
    except OSError as e:
        print(f"[scans] could not remove upload {path}: {e}")


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="List scans, newest first",
    operation_id="listHistory",
    responses={401: {"model": ErrorDetail, "description": "Missing or invalid X-API-Key"}},
)
async def history(
    verdict: str | None = Query(None, pattern="^(HEALTHY|INFECTED)$"),
    limit: int = Query(100, ge=1, le=500),
):
    scans = await run_in_threadpool(store.list_scans, verdict, limit)
    return {"scans": scans, "count": len(scans)}


@router.get(
    "/scan/{scan_id}",
    response_model=ScanResult,
    summary="Get one scan by id",
    operation_id="getScan",
    responses={
        401: {"model": ErrorDetail, "description": "Missing or invalid X-API-Key"},
        404: {"model": ErrorDetail, "description": "No scan with this id"},
    },
)
async def scan_detail(scan_id: str):
    scan = await run_in_threadpool(store.get_scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return scan


@router.get(
    "/stats",
    response_model=Stats,
    summary="Aggregate scan counts",
    operation_id="getStats",
    responses={401: {"model": ErrorDetail, "description": "Missing or invalid X-API-Key"}},
)
async def stats():
    return await run_in_threadpool(store.stats)


@router.delete(
    "/scan/{scan_id}",
    response_model=DeleteOneResult,
    summary="Delete one scan and its stored photo",
    operation_id="deleteScan",
    dependencies=[Depends(require_admin)],
    responses={
        401: {"model": ErrorDetail, "description": "Missing/invalid X-API-Key, or (if configured) X-Admin-Token"},
        404: {"model": ErrorDetail, "description": "No scan with this id"},
    },
)
async def delete_scan(scan_id: str):
    scan = await run_in_threadpool(store.delete_scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail="scan not found")
    _remove_upload(scan.get("imageUrl"))
    return {"deleted": True, "id": scan_id}


@router.delete(
    "/scans",
    response_model=DeleteManyResult,
    summary="Bulk delete scans (and their photos)",
    operation_id="deleteScans",
    dependencies=[Depends(require_admin)],
    responses={
        400: {"model": ErrorDetail, "description": "No filters given and all=true not passed"},
        401: {"model": ErrorDetail, "description": "Missing/invalid X-API-Key, or (if configured) X-Admin-Token"},
    },
)
async def delete_scans(
    verdict: str | None = Query(None, pattern="^(HEALTHY|INFECTED)$"),
    before: str | None = Query(None, description="ISO datetime; deletes scans created before this"),
    confirm_all: bool = Query(False, alias="all", description="Required (=true) to delete with no filters at all"),
):
    if verdict is None and before is None and not confirm_all:
        raise HTTPException(
            status_code=400,
            detail="refusing to delete everything without filters; pass verdict/before, or all=true to confirm",
        )
    deleted = await run_in_threadpool(store.delete_scans, verdict, before)
    for scan in deleted:
        _remove_upload(scan.get("imageUrl"))
    return {"deleted": len(deleted)}
