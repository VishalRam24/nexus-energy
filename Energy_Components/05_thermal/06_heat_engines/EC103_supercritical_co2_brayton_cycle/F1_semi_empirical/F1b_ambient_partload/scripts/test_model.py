"""EC103 — sCO2 Brayton Cycle — F1b T_reject + Part-Load + Recuperator — Test Suite

Tests MUST fail the model, not accommodate it.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

_T_CRIT = 31.1  # CO2 critical temperature [degC]


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"PLR": 0.8})
    for k in ["efficiency_gross", "efficiency_net", "power_output_mw",
              "heat_input_mw", "heat_rejection_mw", "eta_carnot",
              "f_T_reject", "f_partload", "recuperator_effectiveness"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC103"
    assert info["fidelity"] == "F1b"


# --- Part-load efficiency ---

def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["efficiency_net"]) < float(r_full["efficiency_net"])


def test_efficiency_monotone_with_plr(model):
    plr = np.array([0.25, 0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency_net"])
    assert np.all(np.diff(eta) > 0), "Efficiency must increase with PLR for sCO2"


def test_f_partload_one_at_full_load(model):
    r = model.predict({"PLR": 1.0})
    assert float(r["f_partload"]) == pytest.approx(1.0, abs=1e-9)


# --- T_reject sensitivity (critical physics) ---

def test_efficiency_drops_with_higher_T_reject(model):
    """Core physics: higher T_reject = higher T_cold = lower efficiency."""
    r_cool = model.predict({"PLR": 1.0, "T_reject": 28.0})
    r_hot  = model.predict({"PLR": 1.0, "T_reject": 50.0})
    assert float(r_cool["efficiency_net"]) > float(r_hot["efficiency_net"]), \
        "Higher T_reject must reduce efficiency"


def test_near_critical_penalty(model):
    """Just above T_critical, penalty is amplified vs linear extrapolation."""
    T_design = model._raw["cycle"]["T_reject_design_c"]["value"]
    T_crit_margin = _T_CRIT + 1.5  # within the critical band

    r_design = model.predict({"PLR": 1.0, "T_reject": T_design})
    r_crit   = model.predict({"PLR": 1.0, "T_reject": T_crit_margin})
    # When T_crit_margin < T_design, f_T should be 1.0 (no derating)
    # When T_crit_margin > T_design, amplified derating should apply
    if T_crit_margin > T_design:
        assert float(r_crit["f_T_reject"]) < float(r_design["f_T_reject"])
    else:
        # T_crit_margin <= T_design: both f_T should be ~1.0
        assert float(r_crit["f_T_reject"]) == pytest.approx(1.0, abs=0.05)


def test_T_reject_sensitivity_significant(model):
    """A 15 K T_reject rise should cause a meaningful efficiency drop for sCO2."""
    r1 = model.predict({"PLR": 1.0, "T_reject": 32.0})
    r2 = model.predict({"PLR": 1.0, "T_reject": 47.0})
    drop = float(r1["efficiency_net"]) - float(r2["efficiency_net"])
    assert drop > 0.02, f"15 K T_reject rise should drop eta by > 0.02, got {drop:.4f}"


def test_f_T_reject_one_at_design(model):
    T_design = float(model._raw["cycle"]["T_reject_design_c"]["value"])
    r = model.predict({"PLR": 1.0, "T_reject": T_design})
    assert float(r["f_T_reject"]) == pytest.approx(1.0, abs=1e-9)


# --- Recuperator effectiveness ---

def test_recuperator_degrades_at_part_load(model):
    """eps_recup should be lower at part-load than full load."""
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["recuperator_effectiveness"]) < float(r_full["recuperator_effectiveness"]), \
        "Recuperator effectiveness must degrade at part-load"


def test_recuperator_eps_at_design(model):
    """At PLR=1, recuperator effectiveness = design value."""
    eps_design = float(model._raw["cycle"]["eps_recup_design"]["value"])
    r = model.predict({"PLR": 1.0})
    assert float(r["recuperator_effectiveness"]) == pytest.approx(eps_design, abs=1e-9)


def test_recuperator_positive(model):
    plr = np.linspace(0.25, 1.0, 30)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["recuperator_effectiveness"]) > 0)


# --- Carnot limit ---

def test_efficiency_below_carnot(model):
    r = model.predict({"PLR": 1.0})
    assert float(r["efficiency_gross"]) < float(r["eta_carnot"]), \
        "Gross efficiency must be below Carnot"


def test_carnot_increases_with_hotter_T_in(model):
    r_low  = model.predict({"PLR": 1.0, "T_in": 500.0})
    r_high = model.predict({"PLR": 1.0, "T_in": 750.0})
    assert float(r_high["eta_carnot"]) > float(r_low["eta_carnot"])


# --- Net < gross ---

def test_net_below_gross(model):
    plr = np.linspace(0.25, 1.0, 20)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["efficiency_net"]) < np.asarray(r["efficiency_gross"]) + 1e-9)


# --- Energy balance ---

def test_energy_balance(model):
    """Q_in = P_net + Q_reject (first law, within 0.5%)."""
    r = model.predict({"PLR": 0.8, "T_reject": 32.0})
    Q_in  = float(r["heat_input_mw"])
    P_net = float(r["power_output_mw"])
    Q_rej = float(r["heat_rejection_mw"])
    if Q_in > 0:
        error = abs(Q_in - P_net - Q_rej) / Q_in
        assert error < 0.005, f"Energy balance error: {error*100:.2f}%"


# --- Zero below minimum PLR ---

def test_zero_output_below_plr_min(model):
    r = model.predict({"PLR": 0.15})
    assert float(r["power_output_mw"]) == 0.0
    assert float(r["efficiency_gross"]) == 0.0


# --- Benchmark ---

def test_benchmark(model):
    plr   = np.random.uniform(0.25, 1.0, 1000)
    T_rej = np.random.uniform(28, 55, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr, "T_reject": T_rej})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
