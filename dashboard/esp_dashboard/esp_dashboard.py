"""Reflex app entry point for the esp_data dashboard.

The landing page surfaces headline aggregate stats from the precomputed
DuckDB file. The taxonomy sunburst (next step) and per-dataset views
will be added as additional pages.
"""

import reflex as rx
from fastapi import FastAPI

from esp_dashboard.api import router as api_router
from esp_dashboard.dataset_page import dataset_detail
from esp_dashboard.dataset_state import DatasetState
from esp_dashboard.state import LandingState

# Curated datasets surfaced on the landing page as link cards. Order
# matters — first one is featured most prominently in the grid.
_CURATED_DATASETS: tuple[tuple[str, str], ...] = (
    ("inaturalist", "iNaturalist"),
    ("xeno-canto", "Xeno-Canto"),
    ("insectset_459", "InsectSet 459"),
    ("watkins", "Watkins (marine mammals)"),
)

fastapi_app = FastAPI(title="esp-dashboard")
fastapi_app.include_router(api_router)


def _stat_card(label: str, value: rx.Var | str, accent: str = "grass") -> rx.Component:
    """Render one stat tile.

    Parameters
    ----------
    label : str
        Caption shown below the value (e.g. ``"recordings"``).
    value : rx.Var | str
        The headline value, typically bound to a `LandingState` field.
    accent : str, default="grass"
        Radix color name used for the value text.

    Returns
    -------
    rx.Component
        A stat card component with consistent typography.
    """
    return rx.card(
        rx.vstack(
            rx.text(value, size="8", weight="bold", color_scheme=accent),
            rx.text(label, size="2", color_scheme="gray"),
            spacing="1",
            align="start",
        ),
        size="2",
        variant="surface",
    )


def index() -> rx.Component:
    """Landing page — aggregate stats across all public ESP datasets.

    Returns
    -------
    rx.Component
        The full landing page tree.
    """
    return rx.container(
        rx.vstack(
            rx.vstack(
                rx.heading("ESP-Data", size="9"),
                rx.text(
                    "Bioacoustic datasets, unified.",
                    size="5",
                    color_scheme="gray",
                ),
                spacing="2",
                align="start",
            ),
            rx.cond(
                LandingState.error != "",
                rx.callout(
                    LandingState.error,
                    icon="triangle_alert",
                    color_scheme="red",
                    size="2",
                ),
                rx.fragment(),
            ),
            rx.heading("At a glance", size="5", margin_top="6"),
            rx.grid(
                _stat_card("datasets", LandingState.n_datasets),
                _stat_card("recordings", LandingState.n_recordings),
                _stat_card("hours of audio", LandingState.total_hours),
                columns="3",
                spacing="3",
                width="100%",
            ),
            rx.heading("Taxonomic coverage", size="5", margin_top="6"),
            rx.grid(
                _stat_card("phyla", LandingState.n_phyla, accent="iris"),
                _stat_card("classes", LandingState.n_classes, accent="iris"),
                _stat_card("orders", LandingState.n_orders, accent="iris"),
                _stat_card("families", LandingState.n_families, accent="iris"),
                columns="4",
                spacing="3",
                width="100%",
            ),
            rx.heading("Tree of life", size="5", margin_top="6"),
            rx.text(
                "Click a wedge to drill in. Sized by recording count.",
                size="2",
                color_scheme="gray",
            ),
            rx.center(
                rx.plotly(data=LandingState.sunburst, width="900px", height="900px"),
                width="100%",
                margin_top="2",
            ),
            rx.heading("Explore datasets", size="5", margin_top="6"),
            rx.grid(
                *[
                    rx.link(
                        rx.card(
                            rx.vstack(
                                rx.heading(label, size="4"),
                                rx.text(
                                    "metadata · 10 samples · spectrograms",
                                    size="2",
                                    color_scheme="gray",
                                ),
                                spacing="1",
                                align="start",
                            ),
                            size="2",
                            variant="surface",
                            _hover={"background_color": "var(--gray-a3)"},
                        ),
                        href=f"/datasets/{slug}",
                        text_decoration="none",
                        color="inherit",
                    )
                    for slug, label in _CURATED_DATASETS
                ],
                columns="2",
                spacing="3",
                width="100%",
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="4",
        padding_y="6",
    )


app = rx.App(
    theme=rx.theme(appearance="dark", accent_color="grass", radius="medium"),
    api_transformer=fastapi_app,
)
app.add_page(
    index,
    route="/",
    title="ESP-Data Dashboard",
    on_load=LandingState.load_stats,
)
app.add_page(
    dataset_detail,
    route="/datasets/[name]",
    title="ESP-Data · Dataset",
    on_load=DatasetState.load,
)
