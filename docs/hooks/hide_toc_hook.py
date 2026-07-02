"""MkDocs hook to hide the TOC on notebook tutorial pages."""

from mkdocs.structure.pages import Page


def on_page_context(context, page: Page, **kwargs):
    if page.file.src_path.startswith("tutorials/") and page.file.src_path.endswith(".ipynb"):
        page.meta["hide"] = ["toc"]
    return context
