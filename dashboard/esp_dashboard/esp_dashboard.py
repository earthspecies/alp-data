"""Reflex app entry point for the esp_data dashboard.

This is the scaffold (step 1a) — a single landing page placeholder.
Real panels (aggregate stats, sunburst, per-dataset views, NL-SQL)
will be added in subsequent steps.
"""

import reflex as rx
from fastapi import FastAPI

from esp_dashboard.api import router as api_router

fastapi_app = FastAPI(title="esp-dashboard")
fastapi_app.include_router(api_router)


def index() -> rx.Component:
    """Landing page placeholder.

    Returns
    -------
    rx.Component
        A minimal hero with title and status text. Replaced in step 3
        with aggregate stats and the taxonomy sunburst.
    """
    return rx.center(
        rx.vstack(
            rx.heading("ESP-Data Dashboard", size="9"),
            rx.text(
                "Scaffold online. Aggregate stats and per-dataset views coming in the next steps.",
                size="4",
                color_scheme="gray",
            ),
            rx.link(
                rx.button("Health check", variant="soft"),
                href="/api/health",
                is_external=True,
            ),
            spacing="5",
            align="center",
        ),
        height="100vh",
    )


app = rx.App(
    theme=rx.theme(appearance="dark", accent_color="grass", radius="medium"),
    api_transformer=fastapi_app,
)
app.add_page(index, route="/", title="ESP-Data Dashboard")
