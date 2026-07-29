"""EC217 — TEC — F1b Temperature-Dependent Properties — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    for k in ["Q_cold_W", "Q_hot_W", "W_input_W", "COP", "COP_max_theoretical",
              "T_min_achievable_K", "ZT_avg", "V_module_V"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC217"
    assert info["fidelity"] == "F1b"


def test_energy_balance(model):
    """First law: Q_hot = Q_cold + W_input (when Q_cold > 0)."""
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    Q_c = float(np.atleast_1d(r["Q_cold_W"])[0])
    Q_h = float(np.atleast_1d(r["Q_hot_W"])[0])
    W = float(np.atleast_1d(r["W_input_W"])[0])
    # Thomson correction means Q_hot != Q_cold + W exactly, but within 5%
    assert abs(Q_h - (Q_c + W)) < 0.05 * abs(Q_h), \
        f"Energy balance violation: Q_hot={Q_h:.3f}, Q_cold+W={Q_c+W:.3f}"


def test_cop_positive_at_low_current(model):
    """At optimal low current, COP must be positive."""
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 2.0})
    COP = float(np.atleast_1d(r["COP"])[0])
    assert COP >= 0.0, f"COP = {COP:.4f} is negative"


def test_cop_below_carnot(model):
    """COP must be below Carnot COP = T_cold / (T_hot - T_cold)."""
    T_c, T_h = 263.15, 308.15
    r = model.predict({"T_cold_K": T_c, "T_hot_K": T_h, "I_A": 2.0})
    COP = float(np.atleast_1d(r["COP"])[0])
    COP_carnot = T_c / (T_h - T_c)
    assert COP <= COP_carnot * 1.01, f"COP={COP:.4f} > Carnot={COP_carnot:.4f}"


def test_q_cold_increases_with_current_at_low_I(model):
    """Q_cold should increase with current before reaching optimum."""
    Qs = []
    for I in [0.5, 1.0, 2.0]:
        r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": I})
        Qs.append(float(np.atleast_1d(r["Q_cold_W"])[0]))
    assert all(Qs[i] <= Qs[i + 1] for i in range(len(Qs) - 1)), \
        f"Q_cold not increasing with I at low currents: {Qs}"


def test_q_cold_decreases_at_high_current(model):
    """At very high current, Joule heating dominates and Q_cold decreases."""
    r_med = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    r_high = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 15.0})
    Q_med = float(np.atleast_1d(r_med["Q_cold_W"])[0])
    Q_high = float(np.atleast_1d(r_high["Q_cold_W"])[0])
    assert Q_high < Q_med, f"Q_cold should decrease at very high I: Q_med={Q_med:.3f}, Q_high={Q_high:.3f}"


def test_w_input_positive(model):
    """Input power must be positive."""
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    W = float(np.atleast_1d(r["W_input_W"])[0])
    assert W > 0, f"W_input = {W:.4f} is not positive"


def test_zt_reasonable(model):
    """ZT_avg should be 0.3-1.5 for Bi2Te3 in typical range."""
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    ZT = float(np.atleast_1d(r["ZT_avg"])[0])
    assert 0.1 < ZT < 2.0, f"ZT_avg = {ZT:.4f}"


def test_t_min_below_t_cold(model):
    """Minimum achievable temperature must be <= T_cold (TEC can cool lower)."""
    T_c, T_h = 263.15, 308.15
    r = model.predict({"T_cold_K": T_c, "T_hot_K": T_h, "I_A": 3.0})
    T_min = float(np.atleast_1d(r["T_min_achievable_K"])[0])
    assert T_min <= T_c + 1.0, f"T_min={T_min:.2f} K should be <= T_cold={T_c:.2f} K"


def test_alpha_temperature_dependence(model):
    """Seebeck coefficient must decrease at higher temperature (a1 < 0)."""
    a_300 = model._model.alpha(300.0)
    a_400 = model._model.alpha(400.0)
    assert float(a_400) < float(a_300), "alpha should decrease with T"


def test_sigma_temperature_dependence(model):
    """Electrical conductivity must decrease with T (c1 < 0)."""
    s_300 = model._model.sigma_electrical(300.0)
    s_400 = model._model.sigma_electrical(400.0)
    assert float(s_400) < float(s_300), "sigma should decrease with T"


def test_voltage_positive(model):
    """Module terminal voltage must be positive."""
    r = model.predict({"T_cold_K": 263.15, "T_hot_K": 308.15, "I_A": 3.0})
    V = float(np.atleast_1d(r["V_module_V"])[0])
    assert V > 0, f"V_module = {V:.4f} V"


def test_benchmark(model):
    np.random.seed(42)
    T_colds = np.random.uniform(240, 290, 100)
    T_hots = T_colds + np.random.uniform(20, 60, 100)
    Is = np.random.uniform(1.0, 8.0, 100)
    start = time.perf_counter()
    for T_c, T_h, I in zip(T_colds, T_hots, Is):
        model.predict({"T_cold_K": float(T_c), "T_hot_K": float(T_h), "I_A": float(I)})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 5.0
