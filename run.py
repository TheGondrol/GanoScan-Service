"""Dev entrypoint: `python run.py` (auto-reload). For production use uvicorn
directly, e.g. `uvicorn app.main:app --host 0.0.0.0 --port 5005 --workers 2`."""

import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
