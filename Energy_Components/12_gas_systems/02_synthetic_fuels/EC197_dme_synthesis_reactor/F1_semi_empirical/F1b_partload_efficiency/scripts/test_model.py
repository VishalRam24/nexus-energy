"""EC197 — DME Synthesis Reactor — F1b Part-Load + Thermal — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_set": 260.0, "pressure_bar": 40.0})
    for k in ["co_conversion", "effective_temperature_C", "selectivity_dme",
              "dme_production_mol_s", "meoh_slip_mol_s",
              "heat_recovery_kW", "energy_efficiency", "deactivation_factor"]:
        assert k in r, f"Missing: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC197"
    assert info["fidelity"] == "F1b"


def test_conversion_at_design_positive(model):
    X = float(model.model.conversion(260.0, 40.0, plr=1.0))
    assert X > 0.4


def test_effective_temperature_drops_at_part_load(model):
    T_full = float(model.model._effective_temperature(260.0, 1.0))
    T_half = float(model.model._effective_temperature(260.0, 0.5))
    assert T_half < T_full


def test_conversion_decreases_at_part_load(model):
    X_full = float(model.model.conversion(260.0, 40.0, plr=1.0))
    X_half = float(model.model.conversion(260.0, 40.0, plr=0.5))
    assert X_half < X_full


def test_selectivity_decreases_at_part_load(model):
    """Lower PLR → lower T → less dehydration activity → lower DME selectivity."""
    S_full = float(model.model.selectivity_dme(260.0, plr=1.0))
    S_half = float(model.model.selectivity_dme(260.0, plr=0.5))
    assert S_half <= S_full


def test_meoh_slip_positive(model):
    meoh = float(model.model.meoh_slip_mol_s(260.0, 40.0, plr=0.5))
    assert meoh >= 0.0


def test_meoh_plus_dme_production_consistent(model):
    """DME + MeOH products < n_CO_in (some CO unconverted)."""
    n = 1.0
    dme  = float(model.model.dme_production_mol_s(260.0, 40.0, plr=1.0, n_co_in=n))
    meoh = float(model.model.meoh_slip_mol_s(260.0, 40.0, plr=1.0, n_co_in=n))
    # 2 CO → 1 DME; X*n CO converted total; DME uses X*n*S/2 mol DME equivalent CO
    # MeOH uses X*n*(1-S) mol CO
    # CO accounted = 2*dme + meoh <= n (can be less if X<1)
    co_accounted = 2.0 * dme + meoh
    assert co_accounted <= n + 0.01, f"CO balance: {co_accounted:.4f} > n_CO={n}"


def test_deactivation_increases_with_hours(model):
    d0 = float(model.model.deactivation_factor(0.0))
    d5k = float(model.model.deactivation_factor(5000.0))
    assert d0 > d5k


def test_deactivation_cap(model):
    """Max deactivation capped at 30%."""
    d = float(model.model.deactivation_factor(100000.0))
    assert d >= 0.70


def test_heat_recovery_positive(model):
    Q = float(model.model.heat_recovery_kW(260.0, 40.0, plr=1.0))
    assert Q > 0.0


def test_energy_efficiency_bounded(model):
    eta = float(model.model.energy_efficiency(260.0, 40.0))
    assert 0.0 < eta <= 1.0


def test_vectorized(model):
    plr = np.linspace(0.2, 1.0, 20)
    r = model.predict({"T_set": 260.0, "pressure_bar": 40.0, "plr": plr})
    assert r["co_conversion"].shape == (20,)


def test_benchmark(model):
    T = np.random.uniform(220, 300, 1000)
    start = time.perf_counter()
    model.predict({"T_set": T, "pressure_bar": 40.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0
