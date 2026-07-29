"""``nexus_energy.components`` sub-package.

Contains:
  * ``registry`` — 223 F0 component templates. Depends on ``nexus_energy.core``
    so it is lazy-loaded to avoid a circular import during ``core`` init.
  * ``thermal`` — constraint builders for tight UC, PWL heat rate,
    must-run, regulation reserves. Safe to import eagerly.
"""

from .thermal import (
    add_must_run,
    add_regulation_reserve_vars,
    build_pwl_heat_rate,
    build_three_bin_uc,
)

# Names exported from the registry module. Loaded lazily via PEP 562
# module __getattr__ so that ``core`` can import us without triggering
# the registry's ``from nexus_energy.core import ...`` while core is
# still being initialised.
_LAZY_REGISTRY_NAMES = {
    "ComponentTemplate",
    "ComponentRegistry",
    "registry",
    "add_component",
}


def __getattr__(name: str):
    if name in _LAZY_REGISTRY_NAMES:
        from . import registry as _reg
        val = getattr(_reg, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'nexus_energy.components' has no attribute {name!r}")


__all__ = [
    "add_must_run",
    "add_regulation_reserve_vars",
    "build_pwl_heat_rate",
    "build_three_bin_uc",
    "ComponentTemplate",
    "ComponentRegistry",
    "registry",
    "add_component",
]
