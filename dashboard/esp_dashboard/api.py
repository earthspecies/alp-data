"""FastAPI routes mounted onto the Reflex app.

The Reflex `App` exposes a FastAPI instance at `app.api`. We attach
backend endpoints here so the frontend (Reflex/React) can call them
via HTTP. During scaffolding only a health check is provided.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by Cloud Run and local smoke tests.

    Returns
    -------
    dict[str, str]
        A small JSON payload with a `status` key.
    """
    return {"status": "ok"}
