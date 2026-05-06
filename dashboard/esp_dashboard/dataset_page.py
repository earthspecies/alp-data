"""Per-dataset detail page (``/datasets/[name]``).

Renders the metadata header and the 10-sample audio + spectrogram
player. Charts (class distribution, duration histogram) are added in
step 5c.
"""

from __future__ import annotations

import reflex as rx

from esp_dashboard.dataset_state import DatasetState


def _sample_card(sample: rx.Var) -> rx.Component:
    """Render one sample (spectrogram on top, audio + label below).

    Parameters
    ----------
    sample : rx.Var
        A row from `DatasetState.samples` containing keys ``label``,
        ``audio_url``, ``spec_url``, ``duration``, ``license``.

    Returns
    -------
    rx.Component
        The card sub-tree.
    """
    return rx.card(
        rx.vstack(
            rx.image(
                src=sample["spec_url"],
                width="100%",
                border_radius="6px",
            ),
            rx.hstack(
                rx.text(sample["label"], weight="bold", size="2"),
                rx.spacer(),
                rx.badge(sample["duration"], color_scheme="gray", variant="soft"),
                width="100%",
                align="center",
            ),
            rx.audio(
                src=sample["audio_url"],
                width="100%",
                height="40px",
            ),
            rx.cond(
                sample["license"] != "",
                rx.text(sample["license"], size="1", color_scheme="gray"),
                rx.fragment(),
            ),
            spacing="2",
            align="start",
            width="100%",
        ),
        size="2",
        variant="surface",
    )


def _metadata_header() -> rx.Component:
    """Render the dataset header with stats badges.

    Returns
    -------
    rx.Component
        A heading + horizontal row of metadata badges.
    """
    return rx.vstack(
        rx.heading(DatasetState.title, size="8"),
        rx.hstack(
            rx.badge(
                rx.text("v"),
                DatasetState.version,
                color_scheme="gray",
                variant="surface",
            ),
            rx.badge(
                DatasetState.license_text,
                color_scheme="iris",
                variant="soft",
            ),
            spacing="2",
            wrap="wrap",
        ),
        rx.grid(
            rx.card(
                rx.vstack(
                    rx.text(DatasetState.n_recordings, size="6", weight="bold"),
                    rx.text("recordings", size="2", color_scheme="gray"),
                    spacing="0",
                ),
                size="1",
                variant="surface",
            ),
            rx.card(
                rx.vstack(
                    rx.text(DatasetState.total_hours, size="6", weight="bold"),
                    rx.text("hours", size="2", color_scheme="gray"),
                    spacing="0",
                ),
                size="1",
                variant="surface",
            ),
            rx.card(
                rx.vstack(
                    rx.text(DatasetState.n_species, size="6", weight="bold"),
                    rx.text("species", size="2", color_scheme="gray"),
                    spacing="0",
                ),
                size="1",
                variant="surface",
            ),
            rx.card(
                rx.vstack(
                    rx.text(DatasetState.n_families, size="6", weight="bold"),
                    rx.text("families", size="2", color_scheme="gray"),
                    spacing="0",
                ),
                size="1",
                variant="surface",
            ),
            columns="4",
            spacing="3",
            width="100%",
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def dataset_detail() -> rx.Component:
    """Top-level component for ``/datasets/[name]``.

    Returns
    -------
    rx.Component
        The full detail page tree.
    """
    return rx.container(
        rx.vstack(
            rx.link(
                rx.text("← all datasets", size="2"),
                href="/",
                color_scheme="gray",
            ),
            rx.cond(
                DatasetState.error != "",
                rx.callout(
                    DatasetState.error,
                    icon="triangle_alert",
                    color_scheme="red",
                ),
                _metadata_header(),
            ),
            rx.cond(
                DatasetState.family_chart.length() > 0,
                rx.vstack(
                    rx.heading(
                        "Family distribution",
                        size="5",
                        margin_top="6",
                    ),
                    rx.text(
                        "Top families by recording count.",
                        size="2",
                        color_scheme="gray",
                    ),
                    rx.recharts.bar_chart(
                        rx.recharts.bar(data_key="value", fill="#7c5cff"),
                        rx.recharts.x_axis(
                            data_key="name",
                            angle=-40,
                            text_anchor="end",
                            interval=0,
                            height=200,
                        ),
                        rx.recharts.y_axis(),
                        rx.recharts.cartesian_grid(stroke_dasharray="3 3"),
                        rx.recharts.graphing_tooltip(),
                        data=DatasetState.family_chart,
                        width="100%",
                        height=480,
                    ),
                    rx.cond(
                        DatasetState.family_other_count > 0,
                        rx.text(
                            "+ ",
                            DatasetState.family_other_count.to_string(),
                            " more families · ",
                            DatasetState.family_other_total.to_string(),
                            " recordings (long tail, not shown)",
                            size="2",
                            color_scheme="gray",
                        ),
                        rx.fragment(),
                    ),
                    spacing="2",
                    align="start",
                    width="100%",
                ),
                rx.fragment(),
            ),
            rx.heading(
                "Listen — 10 random samples",
                size="5",
                margin_top="6",
            ),
            rx.cond(
                DatasetState.samples.length() > 0,
                rx.grid(
                    rx.foreach(DatasetState.samples, _sample_card),
                    columns="2",
                    spacing="3",
                    width="100%",
                ),
                rx.text(
                    "No samples rendered yet for this dataset.",
                    size="2",
                    color_scheme="gray",
                ),
            ),
            spacing="3",
            align="start",
            width="100%",
        ),
        size="4",
        padding_y="6",
    )
