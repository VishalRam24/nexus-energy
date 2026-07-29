"""Phase 22 — Subsystem composability (fidelity-as-decision, Paper 5).

Validates the thin orchestration layer in
``nexus_energy.components.composition.Subsystem``:

  (a) a valid power→H2→H2store→H2→power chain builds + optimises to optimal on a
      tiny instance, with auto-created intermediate buses and per-component
      fidelity metadata;
  (b) a carrier-mismatched chain raises a clear ``CarrierMismatchError``;
  (c) ``validate_only=True`` detects the mismatch *without* mutating the system.

Chain EC ids used (from the live registry):
    EC008 PEM Electrolyser   electricity → hydrogen   (converter)
    EC012 Compressed H2 Stor. hydrogen   → hydrogen   (storage)
    EC001 PEM Fuel Cell       hydrogen   → electricity (converter)
"""

from __future__ import annotations

import numpy as np
import pytest

import nexus_energy as ne
from nexus_energy.components.composition import (
    Subsystem,
    CarrierMismatchError,
    FIDELITY_LEVELS,
    DEFAULT_FIDELITY,
)


# ---------------------------------------------------------------------------
# (a) valid chain: builds, wires intermediate buses, optimises to optimal
# ---------------------------------------------------------------------------

class TestValidChainBuildsAndOptimises:

    def test_power_to_h2_to_power_chain(self):
        sys = ne.EnergySystem("subsys_p2h2p")
        sys.set_timesteps(4, dt=1.0)

        elec = sys.add_bus("elec")  # electricity carrier (default)

        # Cheap solar in hours 0,1; expensive backup gas always available.
        sys.add_generator("solar", bus=elec, capacity=200, marginal_cost=1.0,
                          carrier_factor=np.array([1.0, 1.0, 0.0, 0.0]))
        sys.add_generator("gas", bus=elec, capacity=200, marginal_cost=200.0)
        sys.add_load("demand", bus=elec, amount=30.0)

        n_buses_before = len(sys._buses)

        sub = Subsystem("p2h2p")
        sub.chain(["EC008", "EC012", "EC001"],
                  fidelity={"EC008": "F1", "EC001": "F4"})
        built = sub.build(
            sys, base_bus=elec,
            capacity={"EC008": 80, "EC012": 50, "EC001": 60},
        )

        # --- structural assertions ------------------------------------
        # interfaces between consecutive pieces are both hydrogen
        assert built["interfaces"] == ["hydrogen", "hydrogen"]

        # exactly ONE new (hydrogen) bus auto-created; elec reused as terminal
        assert len(sys._buses) == n_buses_before + 1
        carriers = {b.carrier.name for b in built["buses"]}
        assert carriers == {"electricity", "hydrogen"}

        # three components emitted, in order, with recorded fidelity
        comps = built["components"]
        assert [c["ec_id"] for c in comps] == ["EC008", "EC012", "EC001"]
        assert comps[0]["fidelity"] == "F1"     # explicit
        assert comps[1]["fidelity"] == DEFAULT_FIDELITY  # default F0
        assert comps[2]["fidelity"] == "F4"     # explicit

        # converters are Links, storage is a Storage (reused add_component path)
        assert comps[0]["object"] in sys._links
        assert comps[2]["object"] in sys._links
        assert comps[1]["object"] in sys._storages

        # --- it optimises -------------------------------------------
        result = sys.optimise()
        assert result.status == "optimal"
        assert 0 < result.total_cost < 1e9

        # electrolyser should consume cheap solar in hour 0 (build H2 inventory)
        elz_flow = result.link_flow[comps[0]["name"]]
        assert elz_flow[0] > 0.1

    def test_single_component_chain(self):
        """A one-element chain is legal: no interfaces, only the base bus."""
        sys = ne.EnergySystem("single")
        sys.set_timesteps(2, dt=1.0)
        elec = sys.add_bus("elec")

        sub = Subsystem("solo").chain(["EC018"])  # LFP battery (elec storage)
        built = sub.build(sys, base_bus=elec, capacity={"EC018": 10})

        assert built["interfaces"] == []
        assert len(built["components"]) == 1
        assert built["components"][0]["fidelity"] == DEFAULT_FIDELITY
        # no extra bus minted — storage sits on the electricity base bus
        assert built["buses"] == [elec]


# ---------------------------------------------------------------------------
# (b) carrier-mismatched chain raises a clear error
# ---------------------------------------------------------------------------

class TestCarrierMismatchRaises:

    def test_mismatch_raises_clear_error(self):
        # EC008 outputs hydrogen; EC068 (heat pump) inputs electricity → mismatch
        sub = Subsystem("bad").chain(["EC008", "EC068"])
        with pytest.raises(CarrierMismatchError) as exc:
            sub.validate()
        msg = str(exc.value)
        assert "EC008" in msg and "EC068" in msg
        assert "hydrogen" in msg and "electricity" in msg

    def test_mismatch_raises_on_build(self):
        sys = ne.EnergySystem("bad_build")
        elec = sys.add_bus("elec")
        n_before = len(sys._buses)
        sub = Subsystem("bad").chain(["EC008", "EC068"])
        with pytest.raises(CarrierMismatchError):
            sub.build(sys, base_bus=elec)
        # nothing was added before the error surfaced
        assert len(sys._buses) == n_before
        assert len(sys._links) == 0


# ---------------------------------------------------------------------------
# (c) validate_only detects mismatch without building
# ---------------------------------------------------------------------------

class TestValidateOnly:

    def test_validate_only_detects_mismatch(self):
        sys = ne.EnergySystem("vo_bad")
        elec = sys.add_bus("elec")
        sub = Subsystem("vo").chain(["EC008", "EC068"])
        with pytest.raises(CarrierMismatchError):
            sub.build(sys, base_bus=elec, validate_only=True)
        # untouched
        assert len(sys._buses) == 1
        assert len(sys._links) == 0

    def test_validate_only_returns_interfaces_no_build(self):
        sys = ne.EnergySystem("vo_ok")
        elec = sys.add_bus("elec")
        n_before = len(sys._buses)
        sub = Subsystem("vo").chain(["EC008", "EC012", "EC001"])
        out = sub.build(sys, base_bus=elec, validate_only=True)
        assert out == {"interfaces": ["hydrogen", "hydrogen"]}
        # validate_only must not mutate the system
        assert len(sys._buses) == n_before
        assert len(sys._links) == 0
        assert len(sys._storages) == 0


# ---------------------------------------------------------------------------
# fidelity taxonomy surface
# ---------------------------------------------------------------------------

class TestFidelityTaxonomy:

    def test_levels_f0_to_f6(self):
        assert set(FIDELITY_LEVELS) == {f"F{i}" for i in range(7)}
        assert DEFAULT_FIDELITY == "F0"

    def test_unknown_fidelity_rejected(self):
        with pytest.raises(ValueError):
            Subsystem("x").chain(["EC008"], fidelity={"EC008": "F9"})

    def test_fidelity_key_not_in_chain_rejected(self):
        with pytest.raises(ValueError):
            Subsystem("x").chain(["EC008"], fidelity={"EC001": "F1"})
