"""MkDocs hook to hide the TOC on notebook tutorial pages."""

from typing import Any

from mkdocs.structure.pages import Page


def on_page_context(context: dict[str, Any], page: Page, **kwargs: Any) -> dict[str, Any]:
    if page.file.src_path.startswith("tutorials/") and page.file.src_path.endswith(".ipynb"):
        page.meta["hide"] = ["toc"]
    return context
