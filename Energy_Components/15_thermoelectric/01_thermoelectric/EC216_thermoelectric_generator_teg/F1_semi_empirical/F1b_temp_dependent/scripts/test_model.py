"""EC216 — TEG — F1b Temperature-Dependent — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    for k in ["efficiency", "power_density_w_cm2", "zt_average", "voltage_V"]:
        assert k in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC216"
    assert info["fidelity"] == "F1b"


def test_zt_at_reference_temp(model):
    """At T0=300K: ZT = alpha0^2 * sigma0 * T0 / k0 = (200e-6)^2 * 1e5 * 300 / 1.5 = 0.8."""
    zt = model._model.zt_local(300.0)
    expected = (200e-6) ** 2 * 1e5 * 300.0 / 1.5
    assert abs(float(zt) - expected) < 0.05, f"ZT(300K) = {float(zt):.4f}, expected {expected:.4f}"


def test_alpha_decreases_with_temperature(model):
    """Seebeck coefficient should decrease at high T (a1 < 0)."""
    alpha_300 = model._model.alpha(300.0)
    alpha_500 = model._model.alpha(500.0)
    assert float(alpha_500) < float(alpha_300)


def test_k_increases_with_temperature(model):
    """Thermal conductivity increases with T (b1 > 0)."""
    k_300 = model._model.k_thermal(300.0)
    k_500 = model._model.k_thermal(500.0)
    assert float(k_500) > float(k_300)


def test_sigma_decreases_with_temperature(model):
    """Electrical conductivity decreases with T (c1 < 0)."""
    s_300 = model._model.sigma_electrical(300.0)
    s_500 = model._model.sigma_electrical(500.0)
    assert float(s_500) < float(s_300)


def test_efficiency_positive(model):
    """Efficiency must be positive for T_hot > T_cold."""
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    eta = float(np.atleast_1d(r["efficiency"])[0])
    assert eta > 0


def test_efficiency_below_carnot(model):
    """Efficiency must be below Carnot limit."""
    T_h, T_c = 473.15, 303.15
    r = model.predict({"T_hot_K": T_h, "T_cold_K": T_c})
    eta = float(np.atleast_1d(r["efficiency"])[0])
    eta_carnot = 1.0 - T_c / T_h
    assert eta < eta_carnot, f"eta={eta:.4f} >= Carnot={eta_carnot:.4f}"


def test_efficiency_increases_with_dt(model):
    """Larger temperature difference should give higher efficiency."""
    r1 = model.predict({"T_hot_K": 373.15, "T_cold_K": 303.15})
    r2 = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    r3 = model.predict({"T_hot_K": 573.15, "T_cold_K": 303.15})
    etas = [float(np.atleast_1d(r["efficiency"])[0]) for r in [r1, r2, r3]]
    assert all(etas[i] <= etas[i + 1] for i in range(len(etas) - 1)), \
        f"Efficiency not increasing with dT: {etas}"


def test_power_density_positive(model):
    """Power density must be positive."""
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    pd = float(np.atleast_1d(r["power_density_w_cm2"])[0])
    assert pd > 0


def test_power_density_increases_with_dt(model):
    """Power density should increase with temperature difference."""
    pds = []
    for T_h in [373.15, 423.15, 473.15, 523.15]:
        r = model.predict({"T_hot_K": T_h, "T_cold_K": 303.15})
        pds.append(float(np.atleast_1d(r["power_density_w_cm2"])[0]))
    assert all(pds[i] <= pds[i + 1] for i in range(len(pds) - 1)), \
        f"Power density not increasing: {pds}"


def test_voltage_positive(model):
    """Voltage at matched load must be positive."""
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    V = float(np.atleast_1d(r["voltage_V"])[0])
    assert V > 0


def test_zt_average_positive(model):
    """Average ZT must be positive."""
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    zt = float(np.atleast_1d(r["zt_average"])[0])
    assert zt > 0


def test_zt_average_reasonable(model):
    """For Bi2Te3, average ZT should be 0.3-1.5 in typical range."""
    r = model.predict({"T_hot_K": 473.15, "T_cold_K": 303.15})
    zt = float(np.atleast_1d(r["zt_average"])[0])
    assert 0.1 < zt < 2.0, f"ZT_avg = {zt:.4f}, expected 0.3-1.5"


def test_benchmark(model):
    T_hots = np.random.uniform(373, 573, 100)
    T_colds = np.random.uniform(273, 323, 100)
    start = time.perf_counter()
    for T_h, T_c in zip(T_hots, T_colds):
        model.predict({"T_hot_K": float(T_h), "T_cold_K": float(T_c)})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 100 predictions in {elapsed * 1000:.2f} ms")
    assert elapsed < 5.0
