"""
Phase 12 — browser / WASM interop.

Public surface:

- :func:`export_lp_for_browser` — serialise an :class:`EnergySystem`
  (or a snapshot subset) to a JSON-ready dict the in-browser HiGHS-
  WASM runtime can consume.
- :func:`import_result_from_browser` — inverse path: rehydrate a
  browser-side solve into :class:`OptimisationResult`.
- :const:`WASM_SCHEMA_VERSION` — bump when the contract changes.
"""

from __future__ import annotations

from nexus_energy.browser.wasm_bridge import (
    WASM_SCHEMA_VERSION,
    export_lp_for_browser,
    import_result_from_browser,
)

__all__ = [
    "WASM_SCHEMA_VERSION",
    "export_lp_for_browser",
    "import_result_from_browser",
]
