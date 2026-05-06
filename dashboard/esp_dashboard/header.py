"""Shared top-of-page header — ESP logo + wordmark.

Imported by every page so the brand and home link stay consistent.
"""

from __future__ import annotations

import reflex as rx


def app_header() -> rx.Component:
    """Render the ESP logo + ``esp-data`` wordmark.

    The whole header is a link back to ``/``. The wordmark uses the
    ESP serif treatment (PT Serif) so it reads as the brand even when
    the rest of the page uses Poppins for sans-serif body text.

    Returns
    -------
    rx.Component
        A horizontal header sitting at the top of every page.
    """
    return rx.link(
        rx.hstack(
            rx.image(
                src="/esp_logo_white.png",
                height="60px",
                width="auto",
            ),
            rx.heading(
                "esp-data",
                size="9",
                weight="bold",
                font_family='"PT Serif", "Times New Roman", serif',
            ),
            spacing="3",
            align="center",
        ),
        href="/",
        text_decoration="none",
        color="inherit",
    )
