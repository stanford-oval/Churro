"""Sphinx configuration for the Churro OCR documentation site."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tomllib

DOCS_DIR = Path(__file__).resolve().parent
ROOT = DOCS_DIR.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

project_data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
project = "Churro OCR"
author = ", ".join(item["name"] for item in project_data["project"]["authors"])
release = project_data["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "pypi.md"]
source_suffix = {".md": "markdown"}
root_doc = "index"
language = "en"

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"
autoclass_content = "both"

napoleon_google_docstring = False
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"{project} Documentation"
html_favicon = "_static/img/churro.png"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/benchmark-leaderboard.js"]
html_sidebars = {
    "**": ["sidebar-nav-full.html"],
}

html_theme_options = {
    "header_links_before_dropdown": 0,
    "icon_links": [
        {
            "name": "GitHub",
            "url": project_data["project"]["urls"]["Repository"],
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "",
    },
    "navbar_align": "left",
    "navbar_center": [],
    "footer_end": [],
    "footer_start": [],
    "navigation_with_keys": True,
    "search_as_you_type": True,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "show_toc_level": 2,
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "stanford-oval",
    "github_repo": "Churro",
    "github_version": "main",
    "doc_path": "docs",
}


def _copy_build_artifacts(app, exception) -> None:
    if exception is not None:
        return

    benchmark_output_dir = Path(app.outdir) / "_static" / "data"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "benchmark_results.json",
        benchmark_output_dir / "benchmark_results.json",
    )

    shutil.copytree(ROOT / "static", Path(app.outdir) / "static", dirs_exist_ok=True)


def setup(app) -> None:
    app.connect("build-finished", _copy_build_artifacts)
