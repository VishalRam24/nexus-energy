"""Phase 22 — Component composability (fidelity-as-decision, Paper 5).

A *thin orchestration layer* over :func:`nexus_energy.components.add_component`
that turns the 223-template library into composable building blocks.

The core idea (research novelty "component composability"): a user describes a
**serial conversion chain** as a list of EC ids — e.g. electrolyser → H2 store →
fuel cell — and the :class:`Subsystem` orchestrator

  1. *validates* that the carrier interfaces line up
     (``template[k].output_carrier == template[k+1].input_carrier``), raising a
     clear error on a mismatch;
  2. *auto-creates* the intermediate :class:`~nexus_energy.core.Bus` objects with
     the right carrier (registering the carrier first if the system does not know
     it yet, exactly like :func:`sectors.create_temperature_heat_network`);
  3. emits every piece through the existing ``add_component`` — no primitive
     mapping is re-implemented here — wiring converters as ``bus_from → bus_to``
     and generators / storages onto their single bus;
  4. records a per-component **fidelity** choice (F0–F6 taxonomy) as metadata.

Fidelity-as-decision scope note
-------------------------------
The ``fidelity=`` selector is *recorded and forwarded as metadata only*. Full
multi-fidelity physics routing (swapping in F1–F6 surrogate / PDE / PINN models
per component) is **out of scope for this phase** — the API accepts and stores
the choice so downstream phases can route on it, but every component is still
emitted with its F0 (constant-efficiency) behaviour via ``add_component``.

This module does **not** modify ``registry.py``, ``core.py`` or ``__init__.py``;
it only reads ``registry.get`` / ``registry.list_components`` and reuses
``add_component``. Import it explicitly::

    from nexus_energy.components.composition import Subsystem, FIDELITY_LEVELS
"""

from __future__ import annotations

from typing import Optional

from nexus_energy.core import Bus, EnergySystem
from nexus_energy.components.registry import add_component, registry


# F0–F6 fidelity taxonomy (see Energy_Components/CLAUDE.md). F0 is the default
# constant-efficiency model every template currently ships.
FIDELITY_LEVELS: dict[str, str] = {
    "F0": "Empirical (lookup tables / efficiency curves)",
    "F1": "Semi-Empirical (analytical eqs + fitted params)",
    "F2": "Physics — Lumped (0D/1D ODE first-principles)",
    "F3": "Physics — Distributed (PDE spatially-resolved)",
    "F4": "AI Static (MLP/ANN steady-state surrogate)",
    "F5": "AI Dynamic (LSTM/Transformer time-series)",
    "F6": "PINN (Physics-Informed Neural Network)",
}

DEFAULT_FIDELITY = "F0"

# Categories whose template carries a single bus (no separate bus_to port).
_SINGLE_BUS_CATEGORIES = {"generator", "storage"}


class CarrierMismatchError(ValueError):
    """Raised when two consecutive components in a chain do not share a carrier.

    ``template[k].output_carrier`` must equal ``template[k+1].input_carrier`` for
    the chain to be physically wireable.
    """


def _normalise_fidelity(ec_id: str, fidelity: Optional[str]) -> str:
    """Validate / default a fidelity tag for one component."""
    if fidelity is None:
        return DEFAULT_FIDELITY
    f = str(fidelity).upper()
    if f not in FIDELITY_LEVELS:
        raise ValueError(
            f"Unknown fidelity {fidelity!r} for {ec_id}. "
            f"Valid levels: {sorted(FIDELITY_LEVELS)}."
        )
    return f


class Subsystem:
    """Composable serial conversion chain over the component registry.

    Usage::

        sub = Subsystem("p2h2p")
        sub.chain(["EC008", "EC012", "EC001"],          # elec→H2→H2store→H2→elec
                  fidelity={"EC008": "F1"})             # optional per-component
        built = sub.build(system, base_bus=elec_bus,
                          capacity={"EC008": 50, "EC001": 30})

    The serial-chain carrier contract is: for every consecutive pair the upstream
    component's ``output_carrier`` must equal the downstream component's
    ``input_carrier``. Storage components (``input_carrier == output_carrier``)
    sit transparently on the carrier bus they buffer.
    """

    def __init__(self, name: str):
        self.name = name
        self._chain: list[str] = []
        # ec_id (positional key) -> fidelity tag. Keyed by position to allow the
        # same EC id to appear twice in a chain with different fidelities.
        self._fidelity: dict[int, str] = {}

    # -- chain definition ---------------------------------------------------

    def chain(
        self,
        ec_ids: list[str],
        fidelity: Optional[dict[str, str]] = None,
    ) -> "Subsystem":
        """Define the serial conversion chain (ordered list of EC ids).

        Args:
            ec_ids: ordered EC ids, e.g. ``["EC008", "EC012", "EC001"]``.
            fidelity: optional ``{ec_id: "F0".."F6"}`` map. Components not listed
                default to F0. Each id is validated against the registry and each
                level against :data:`FIDELITY_LEVELS`.

        Returns ``self`` for fluent chaining.
        """
        if len(ec_ids) < 1:
            raise ValueError("chain() needs at least one component.")
        # Validate every id exists (registry.get raises KeyError otherwise).
        for ec_id in ec_ids:
            registry.get(ec_id)
        self._chain = list(ec_ids)

        fidelity = fidelity or {}
        unknown = set(fidelity) - set(ec_ids)
        if unknown:
            raise ValueError(
                f"fidelity keys {sorted(unknown)} are not in the chain {ec_ids}."
            )
        self._fidelity = {
            idx: _normalise_fidelity(ec_id, fidelity.get(ec_id))
            for idx, ec_id in enumerate(ec_ids)
        }
        return self

    # -- carrier validation -------------------------------------------------

    def validate(self) -> list[str]:
        """Validate carrier compatibility of the whole chain.

        Returns the ordered list of *interface carriers* between consecutive
        components (length ``len(chain) - 1``). For a single-component chain the
        list is empty. Raises :class:`CarrierMismatchError` on the first
        mismatch with a clear message.
        """
        if not self._chain:
            raise ValueError("No chain defined; call .chain([...]) first.")
        interfaces: list[str] = []
        for k in range(len(self._chain) - 1):
            up_id = self._chain[k]
            dn_id = self._chain[k + 1]
            up = registry.get(up_id)
            dn = registry.get(dn_id)
            if up.output_carrier != dn.input_carrier:
                raise CarrierMismatchError(
                    f"Carrier mismatch in subsystem {self.name!r} between "
                    f"{up_id} ({up.name}) and {dn_id} ({dn.name}): "
                    f"{up_id}.output_carrier={up.output_carrier!r} != "
                    f"{dn_id}.input_carrier={dn.input_carrier!r}. "
                    f"They cannot be wired in series."
                )
            interfaces.append(up.output_carrier)
        return interfaces

    # -- build --------------------------------------------------------------

    def build(
        self,
        system: EnergySystem,
        base_bus: Bus,
        *,
        capacity: Optional[dict[str, float]] = None,
        default_capacity: float = 100.0,
        validate_only: bool = False,
        **per_component_overrides,
    ) -> dict[str, object]:
        """Materialise the chain onto ``system`` starting from ``base_bus``.

        ``base_bus`` feeds the *input_carrier* of the first component. Each
        consecutive interface carrier gets a freshly created intermediate bus
        (carrier registered on the system if absent). The terminal output bus is
        either ``base_bus`` again (if the chain closes back onto the starting
        carrier, e.g. a power→H2→power round trip) or a new bus for the final
        output carrier.

        Args:
            system: target :class:`EnergySystem`.
            base_bus: input bus for the first component.
            capacity: optional ``{ec_id: MW}`` per-component capacity map;
                falls back to ``default_capacity``.
            default_capacity: capacity for components not in ``capacity``.
            validate_only: if True, only run :meth:`validate` and return the
                interface carrier list under ``{"interfaces": [...]}`` — nothing
                is added to ``system``.
            **per_component_overrides: ``{ec_id: {kwarg: val}}`` forwarded to
                ``add_component`` for that component (e.g. ``extendable``,
                ``marginal_cost``, ``energy_to_power_ratio``).

        Returns a dict of created objects: ``"components"`` (ordered list of
        ``{ec_id, name, fidelity, object}`` records), ``"buses"`` (ordered list
        of every bus in the chain, including ``base_bus``), and
        ``"interfaces"`` (interface carrier names).
        """
        interfaces = self.validate()
        if validate_only:
            return {"interfaces": interfaces}

        capacity = capacity or {}
        first = registry.get(self._chain[0])
        if base_bus.carrier.name != first.input_carrier:
            raise CarrierMismatchError(
                f"base_bus carrier {base_bus.carrier.name!r} does not match the "
                f"first component {self._chain[0]} input_carrier "
                f"{first.input_carrier!r}."
            )

        # Bus per carrier boundary. The chain visits these carriers in order:
        #   [in_carrier, interface_0, interface_1, ..., out_carrier]
        # base_bus is reused for its carrier; reuse it again if the chain returns
        # to that carrier (closes a loop), otherwise mint a fresh bus per new
        # carrier boundary.
        bus_for_carrier: dict[str, Bus] = {base_bus.carrier.name: base_bus}
        all_buses: list[Bus] = [base_bus]

        def _bus(carrier: str, tag: str) -> Bus:
            if carrier in bus_for_carrier:
                return bus_for_carrier[carrier]
            if carrier not in system._carriers:
                system.add_carrier(carrier)
            b = system.add_bus(f"{self.name}_{tag}", carrier=carrier)
            bus_for_carrier[carrier] = b
            all_buses.append(b)
            return b

        components: list[dict] = []
        for idx, ec_id in enumerate(self._chain):
            tmpl = registry.get(ec_id)
            cap = capacity.get(ec_id, default_capacity)
            overrides = dict(per_component_overrides.get(ec_id, {}))
            comp_name = f"{self.name}_{idx}_{ec_id}"

            in_bus = _bus(tmpl.input_carrier, f"{tmpl.input_carrier}")
            if tmpl.category in _SINGLE_BUS_CATEGORIES:
                # Generator / storage: single bus on its (in==out) carrier.
                obj = add_component(
                    system, comp_name, ec_id,
                    bus=in_bus, capacity=cap, **overrides,
                )
            else:
                # Converter: input_carrier bus → output_carrier bus.
                out_bus = _bus(tmpl.output_carrier, f"{tmpl.output_carrier}")
                obj = add_component(
                    system, comp_name, ec_id,
                    bus=in_bus, bus_to=out_bus, capacity=cap, **overrides,
                )

            components.append({
                "ec_id": ec_id,
                "name": comp_name,
                "fidelity": self._fidelity[idx],
                "object": obj,
            })

        return {
            "components": components,
            "buses": all_buses,
            "interfaces": interfaces,
        }
