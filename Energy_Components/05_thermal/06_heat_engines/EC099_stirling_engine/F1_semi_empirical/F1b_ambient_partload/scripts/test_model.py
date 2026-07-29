"""EC099 — Stirling Engine — F1b Ambient T_c + Part-Load — Test Suite

Tests MUST fail the model, not accommodate it.
"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


# --- Interface ---

def test_predict_keys(model):
    r = model.predict({"PLR": 0.8})
    for k in ["efficiency_gross", "efficiency_net", "power_output_w",
              "heat_input_w", "heat_rejection_w", "eta_carnot",
              "T_cold_side_c", "f_partload"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC099"
    assert info["fidelity"] == "F1b"


# --- Efficiency drops at part-load ---

def test_efficiency_drops_at_part_load(model):
    r_full = model.predict({"PLR": 1.0})
    r_part = model.predict({"PLR": 0.3})
    assert float(r_part["efficiency_net"]) < float(r_full["efficiency_net"])


def test_efficiency_monotone_with_plr(model):
    plr = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    r = model.predict({"PLR": plr})
    eta = np.asarray(r["efficiency_net"])
    assert np.all(np.diff(eta) > 0), "Efficiency must increase with PLR"


def test_f_partload_one_at_full_load(model):
    r = model.predict({"PLR": 1.0})
    assert float(r["f_partload"]) == pytest.approx(1.0, abs=1e-9)


# --- Ambient T_c dependence ---

def test_warm_ambient_reduces_efficiency(model):
    """Warmer ambient => higher T_c => lower Carnot => lower efficiency."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": -5.0})
    r_warm = model.predict({"PLR": 1.0, "T_ambient": 40.0})
    assert float(r_cold["efficiency_net"]) > float(r_warm["efficiency_net"]), \
        "Cold ambient should give higher efficiency"


def test_T_cold_increases_with_ambient(model):
    """T_c = T_ambient + offset: T_cold must track ambient."""
    r1 = model.predict({"PLR": 1.0, "T_ambient": 10.0})
    r2 = model.predict({"PLR": 1.0, "T_ambient": 35.0})
    assert float(r2["T_cold_side_c"]) > float(r1["T_cold_side_c"]), \
        "Cold-side temperature must rise with ambient"


def test_ambient_effect_significant(model):
    """A 30 K ambient rise should cause measurable efficiency drop."""
    r1 = model.predict({"PLR": 1.0, "T_ambient": 5.0})
    r2 = model.predict({"PLR": 1.0, "T_ambient": 35.0})
    drop = float(r1["efficiency_net"]) - float(r2["efficiency_net"])
    assert drop > 0.01, f"30 K ambient rise should degrade eta by > 0.01, got {drop:.4f}"


# --- T_h dependence ---

def test_higher_T_h_increases_efficiency(model):
    """Higher heater head temperature => higher Carnot => better efficiency."""
    r_low  = model.predict({"PLR": 1.0, "T_hot": 450.0})
    r_high = model.predict({"PLR": 1.0, "T_hot": 750.0})
    assert float(r_high["efficiency_net"]) > float(r_low["efficiency_net"])


# --- Carnot limit ---

def test_efficiency_below_carnot(model):
    r = model.predict({"PLR": 1.0})
    eta_net   = float(r["efficiency_net"])
    eta_carnot = float(r["eta_carnot"])
    # gross < carnot; net < gross < carnot
    assert eta_net < eta_carnot, f"Net eta={eta_net:.3f} must be below Carnot={eta_carnot:.3f}"


def test_carnot_increases_with_cold_ambient(model):
    """Colder ambient => larger T_h/T_c spread => higher Carnot."""
    r_cold = model.predict({"PLR": 1.0, "T_ambient": -10.0})
    r_warm = model.predict({"PLR": 1.0, "T_ambient":  40.0})
    assert float(r_cold["eta_carnot"]) > float(r_warm["eta_carnot"])


# --- Net < gross ---

def test_net_below_gross(model):
    plr = np.linspace(0.2, 1.0, 20)
    r = model.predict({"PLR": plr})
    assert np.all(np.asarray(r["efficiency_net"]) < np.asarray(r["efficiency_gross"]) + 1e-9)


# --- Energy balance ---

def test_energy_balance(model):
    """Q_in = P_net + Q_reject (first law, within 0.5%)."""
    r = model.predict({"PLR": 0.8, "T_ambient": 20.0})
    Q_in  = float(r["heat_input_w"])
    P_net = float(r["power_output_w"])
    Q_rej = float(r["heat_rejection_w"])
    if Q_in > 0:
        error = abs(Q_in - P_net - Q_rej) / Q_in
        assert error < 0.005, f"Energy balance error: {error*100:.2f}%"


# --- Zero below minimum load ---

def test_zero_output_below_plr_min(model):
    r = model.predict({"PLR": 0.10})
    assert float(r["power_output_w"]) == 0.0
    assert float(r["efficiency_gross"]) == 0.0


# --- Benchmark ---

def test_benchmark(model):
    plr  = np.random.uniform(0.2, 1.0, 1000)
    T_a  = np.random.uniform(-10, 40, 1000)
    start = time.perf_counter()
    model.predict({"PLR": plr, "T_ambient": T_a})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
