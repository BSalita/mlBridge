"""Version numbers of key packages, for app footers.

Reads installed distribution metadata, so heavy packages such as torch are
never imported just to show their version. Packages that are not installed
are left out.
"""

from __future__ import annotations

import sys
from importlib import metadata

# (label shown in footer, pip distribution name)
KEY_PACKAGES: tuple[tuple[str, str], ...] = (
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("pyarrow", "pyarrow"),
    ("duckdb", "duckdb"),
    ("numpy", "numpy"),
    ("endplay", "endplay"),
    ("pytorch", "torch"),
    ("scikit-learn", "scikit-learn"),
)


def package_versions(packages: tuple[tuple[str, str], ...] = KEY_PACKAGES) -> dict[str, str]:
    """Return {label: version}, starting with Python, for installed packages only."""
    versions = {"Python": ".".join(map(str, sys.version_info[:3]))}
    for label, dist in packages:
        try:
            versions[label] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


def package_versions_line(packages: tuple[tuple[str, str], ...] = KEY_PACKAGES) -> str:
    """One-line footer text, e.g. 'Python:3.12.4 streamlit:1.40.0 polars:1.20.0 ...'."""
    return " ".join(f"{k}:{v}" for k, v in package_versions(packages).items())
