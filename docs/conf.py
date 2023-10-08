"""Sphinx configuration."""
project = "DANKPY"
author = "Daniel Kadyrov"
copyright = "2023, Daniel Kadyrov"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
