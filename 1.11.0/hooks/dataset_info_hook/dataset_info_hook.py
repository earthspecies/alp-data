"""
MkDocs hook to inject dataset info into documentation.

This hook automatically adds DatasetInfo metadata to each dataset's
documentation by intercepting the HTML generation process.
"""

import logging
import re

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

from alp_data.dataset import DatasetInfo

log = logging.getLogger("mkdocs.plugins.dataset_info_hook")


def format_dataset_info_html(info: "DatasetInfo") -> str:
    """Format a dataset's DatasetInfo as HTML.

    Parameters
    ----------
    info : DatasetInfo
        The dataset info object

    Returns
    -------
    str
        Formatted HTML string
    """
    # Format sources
    if isinstance(info.sources, list):
        sources_str = ", ".join(info.sources)
    else:
        sources_str = info.sources

    # Format split paths - limit display if too many
    splits_list = list(info.split_paths.keys())
    if len(splits_list) > 10:
        splits_display = ", ".join(f"<code>{s}</code>" for s in splits_list[:10])
        splits_display += f", ... ({len(splits_list)} total)"
    else:
        splits_display = ", ".join(f"<code>{s}</code>" for s in splits_list)

    # Build HTML
    html = []
    html.append('<details class="dataset-info">')
    html.append("<summary><strong>📊 Dataset Information</strong></summary>")
    html.append('<div class="dataset-info-content">')
    html.append("<table>")
    html.append("<tbody>")
    html.append(f"<tr><td><strong>Name</strong></td><td><code>{info.name}</code></td></tr>")
    html.append(f"<tr><td><strong>Version</strong></td><td><code>{info.version}</code></td></tr>")
    html.append(f"<tr><td><strong>Owner</strong></td><td>{info.owner}</td></tr>")
    html.append(f"<tr><td><strong>License</strong></td><td>{info.license}</td></tr>")
    html.append(f"<tr><td><strong>Sources</strong></td><td>{sources_str}</td></tr>")
    html.append(f"<tr><td><strong>Available Splits</strong></td><td>{splits_display}</td></tr>")
    html.append("</tbody>")
    html.append("</table>")

    if info.description and info.description not in ["[MISSING]", ""]:
        html.append('<div class="dataset-description">')
        html.append("<strong>Description:</strong>")
        html.append(f"<p>{info.description}</p>")
        html.append("</div>")

    if info.changelog and info.changelog != "":
        html.append('<div class="dataset-changelog">')
        html.append("<strong>Changelog:</strong>")
        html.append(f"<p>{info.changelog}</p>")
        html.append("</div>")

    html.append("</div>")
    html.append("</details>")

    return "\n".join(html)


def on_page_content(html: str, page: "Page", config: "MkDocsConfig", files: "Files") -> str:
    """Hook to modify HTML after it's generated.

    This function is called by MkDocs for each page after the HTML is generated.
    If the page is the datasets page, it injects dataset info after each dataset class heading.

    Parameters
    ----------
    html : str
        The HTML content of the page
    page : Page
        The page object
    config : MkDocsConfig
        The MkDocs config
    files : Files
        All files in the documentation

    Returns
    -------
    str
        Modified HTML content
    """
    # Only process the datasets page
    if page.file.src_path != "datasets.md":
        return html

    try:
        # Import here to avoid issues during mkdocs build
        import alp_data.datasets  # noqa: F401 - ensure all datasets are registered
        from alp_data.dataset import _dataset_registry

        # Build a mapping of class names to their info HTML
        dataset_info_map = {}
        for dataset_class in _dataset_registry.values():
            class_name = dataset_class.__name__
            info_html = format_dataset_info_html(dataset_class.info)
            dataset_info_map[class_name] = info_html

        # Pattern to find dataset class sections
        # Looking for <h3 id="alp_data.datasets.ClassName">
        def inject_after_heading(match: re.Match[str]) -> str:
            full_match = match.group(0)
            class_id = match.group(1)  # e.g., "alp_data.datasets.AnimalSpeak"

            # Extract class name from the ID
            if "." in class_id:
                class_name = class_id.split(".")[-1]
            else:
                class_name = class_id

            if class_name in dataset_info_map:
                # Inject the dataset info right after the </h3> tag
                return full_match + "\n" + dataset_info_map[class_name]
            return full_match

        # Match h3 tags with dataset class IDs
        pattern = r'<h3[^>]*id="([^"]*datasets\.([^"]+))"[^>]*>.*?</h3>'
        modified_html = re.sub(pattern, inject_after_heading, html, flags=re.DOTALL)

        log.info(f"Injected dataset info for {len(dataset_info_map)} datasets")
        return modified_html

    except Exception as e:
        log.error(f"Failed to inject dataset info: {e}", exc_info=True)
        return html
