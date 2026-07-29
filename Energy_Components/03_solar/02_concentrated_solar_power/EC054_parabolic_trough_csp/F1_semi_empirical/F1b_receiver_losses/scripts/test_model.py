"""EC054 — Parabolic Trough CSP — F1b Receiver Losses — Test Suite"""

import sys, time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_returns_dict(model):
    r = model.predict({
        "DNI_w_m2": 800.0, "T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0
    })
    for key in ["thermal_output_kw_per_m", "optical_efficiency",
                "thermal_efficiency", "receiver_loss_kw_per_m"]:
        assert key in r


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC054"
    assert info["fidelity"] == "F1b"


def test_thermal_output_positive(model):
    """At high DNI and moderate temps, thermal output should be positive."""
    r = model.predict({
        "DNI_w_m2": 800.0, "T_htf_in_degC": 200.0, "T_htf_out_degC": 300.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 0.0
    })
    assert float(r["thermal_output_kw_per_m"]) > 0


def test_thermal_output_increases_with_dni(model):
    """Higher DNI -> more thermal output."""
    base = {"T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
            "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0}
    G = np.array([400.0, 600.0, 800.0, 1000.0])
    r = model.predict({**base, "DNI_w_m2": G})
    assert np.all(np.diff(r["thermal_output_kw_per_m"]) > 0)


def test_receiver_loss_increases_with_temperature(model):
    """Higher absorber temp -> more receiver loss."""
    T_abs = np.array([150.0, 250.0, 350.0])
    q_loss = model._model.receiver_loss_kw_per_m(T_abs, 25.0)
    assert np.all(np.diff(q_loss) > 0)


def test_iam_unity_at_normal_incidence(model):
    """IAM should be ~1 at theta=0."""
    iam = model._model.iam(0.0)
    assert 0.99 < float(iam) <= 1.0


def test_iam_decreases_with_angle(model):
    """IAM should decrease with incidence angle."""
    theta = np.array([0.0, 20.0, 40.0, 60.0])
    iam = model._model.iam(theta)
    assert np.all(np.diff(iam) < 0)


def test_end_loss_unity_at_normal(model):
    """End loss factor should be 1 at theta=0."""
    f_end = model._model.end_loss_factor(0.0)
    assert float(f_end) == pytest.approx(1.0, abs=0.01)


def test_end_loss_decreases_with_angle(model):
    """End loss factor should decrease with angle."""
    theta = np.array([0.0, 20.0, 40.0, 60.0])
    f_end = model._model.end_loss_factor(theta)
    assert np.all(np.diff(f_end) < 0)


def test_optical_efficiency_range(model):
    """Optical efficiency should be 0.5-0.85 at normal incidence."""
    eta = model._model.optical_efficiency(0.0)
    assert 0.5 < float(eta) < 0.90


def test_zero_dni(model):
    """No sun = no output."""
    r = model.predict({
        "DNI_w_m2": 0.0, "T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0
    })
    assert float(r["thermal_output_kw_per_m"]) == 0.0


def test_receiver_loss_radiation_dominates_high_temp(model):
    """At high temps, radiative loss should be significant."""
    q_low = model._model.receiver_loss_kw_per_m(200.0, 25.0)
    q_high = model._model.receiver_loss_kw_per_m(400.0, 25.0)
    # Radiative ~ T^4, so ratio should be > linear
    ratio = float(q_high) / float(q_low)
    assert ratio > 3.0, f"Loss ratio = {ratio:.1f}, expected strong T^4 dependence"


def test_array_inputs(model):
    G = np.array([400.0, 600.0, 800.0])
    r = model.predict({
        "DNI_w_m2": G, "T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0
    })
    assert r["thermal_output_kw_per_m"].shape == (3,)


def test_benchmark(model):
    G = np.random.uniform(200, 900, 1000)
    start = time.perf_counter()
    model.predict({
        "DNI_w_m2": G, "T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
        "T_ambient_degC": 25.0, "incidence_angle_deg": 15.0
    })
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 predictions in {elapsed*1000:.2f} ms")
    assert elapsed < 1.0
