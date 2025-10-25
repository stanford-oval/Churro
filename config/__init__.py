"""Configuration helpers for Churro.

Expose `get_settings` as the canonical accessor for environment-driven
configuration. Modules should avoid loading `.env` directly and instead
import from this package to retrieve typed snapshots.
"""

from .settings import ChurroSettings, get_settings


__all__ = ["ChurroSettings", "get_settings"]
