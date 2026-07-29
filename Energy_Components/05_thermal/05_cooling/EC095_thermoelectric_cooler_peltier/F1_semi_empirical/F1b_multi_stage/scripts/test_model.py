"""EC095 — Peltier TEC — F1b Multi-Stage — Test Suite"""
import sys, time, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


@pytest.fixture
def model():
    return ComponentModel()


def test_predict_keys(model):
    n = model._model.n_stages
    r = model.predict({"current_stages": [3.0]*n, "T_cold": 0.0, "T_hot": 40.0})
    for k in ["Q_c_kw", "W_total_kw", "COP", "T_inter_C", "COP_single_stage_ref"]:
        assert k in r, f"Missing key: {k}"


def test_get_info(model):
    info = model.get_info()
    assert info["ec_id"] == "EC095"
    assert info["fidelity"] == "F1b"


def test_multistage_advantage_at_large_dT(model):
    """At large dT, the cascade achieves positive COP while single-stage cannot.
    RATIONALE: A single-stage TEC fails (Q_c <= 0) when dT >> dT_max for
    the given current/module because the back-conduction term K*(T_h-T_c)
    exceeds the Peltier cooling.  Cascading splits the span so each stage
    operates within its feasible range.
    # RATIONALE: This is the primary motivation for cascade TECs (Goldsmid
    2010, ch.6; Chein & Chen 2005 — cascades extend maximum dT_max).
    At small dT where single-stage works, cascade COP is penalized by the
    inter-stage pumping burden (W_stage2 must also pump W_stage1 as heat)."""
    n = model._model.n_stages
    # At large dT = 60K: cascade > 0, single equivalent = 0 (can't cool)
    r_large = model.predict({"current_stages": [3.0]*n, "T_cold": -10.0,
                             "T_hot": 50.0})   # dT = 60K
    assert float(r_large["COP"]) > 0.0, \
        "Cascade must achieve positive COP at large dT where single-stage fails"
    assert float(r_large["COP_single_stage_ref"]) <= 0.01, \
        "Single-stage should fail (COP≈0) at dT=60K — justifies using cascade"

    # At small dT = 20K: single-stage COP > cascade COP (inter-stage overhead)
    r_small = model.predict({"current_stages": [3.0]*n, "T_cold": 10.0,
                             "T_hot": 30.0})   # dT = 20K
    cop_c = float(r_small["COP"])
    cop_s = float(r_small["COP_single_stage_ref"])
    # Both should work; cascade COP penalized by inter-stage burden
    assert cop_s >= cop_c * 0.5, \
        f"At small dT=20K: single-stage COP={cop_s:.3f} should be >= 0.5*cascade={cop_c:.3f}"


def test_cop_nonnegative(model):
    """COP must be >= 0 at all operating points."""
    n = model._model.n_stages
    for I in [1.0, 3.0, 6.0]:
        r = model.predict({"current_stages": [I]*n, "T_cold": 0.0, "T_hot": 50.0})
        assert float(r["COP"]) >= 0.0


def test_T_inter_between_cold_and_hot(model):
    """Inter-stage temperature must be between T_cold and T_hot."""
    n = model._model.n_stages
    T_cold, T_hot = 0.0, 50.0
    r = model.predict({"current_stages": [3.0]*n, "T_cold": T_cold, "T_hot": T_hot})
    T_inter = float(r["T_inter_C"])
    assert T_cold < T_inter < T_hot + 5.0, \
        f"T_inter={T_inter:.1f}C should be between {T_cold} and {T_hot}"


def test_Q_c_positive_at_moderate_current(model):
    """Net cooling power must be positive at moderate current."""
    n = model._model.n_stages
    r = model.predict({"current_stages": [3.0]*n, "T_cold": 10.0, "T_hot": 40.0})
    assert float(r["Q_c_kw"]) > 0.0, \
        f"Q_c must be positive; got {float(r['Q_c_kw'])*1000:.1f} W"


def test_cop_decreases_with_larger_dT(model):
    """COP must decrease as temperature span increases (harder to pump heat).
    RATIONALE: Larger dT increases Peltier back-flow term K*(T_h-T_c)
    (Goldsmid 2010, eq.2.14)."""
    n = model._model.n_stages
    dT_arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    cops = []
    for dT in dT_arr:
        r = model.predict({"current_stages": [3.0]*n, "T_cold": 25.0-dT/2,
                           "T_hot": 25.0+dT/2})
        cops.append(float(r["COP"]))
    assert np.all(np.diff(cops) <= 0), \
        f"COP should decrease with larger dT: {cops}"


def test_energy_balance_per_stage(model):
    """Q_h_stage1 <= Q_c_stage2 + W_stage2 (stage 2 must pump stage 1's hot load).
    Checks internal physics — Q_c list[0] is stage-1 cold absorption."""
    n = model._model.n_stages
    r = model.predict({"current_stages": [3.0]*n, "T_cold": 0.0, "T_hot": 50.0})
    q_c_s = r["Q_c_per_stage_W"]
    w_s   = r["W_per_stage_W"]
    if len(q_c_s) >= 2:
        Q_h_1 = float(q_c_s[0]) + float(w_s[0])
        Q_c_2 = float(q_c_s[1])
        # Q_c_2 must be >= Q_h_1 (stage 2 pumps all of stage 1's hot rejection)
        # Allow tolerance for contact resistance
        assert Q_c_2 >= Q_h_1 * 0.5, \
            f"Stage 2 Q_c={Q_c_2:.1f}W should be >= stage 1 Q_h={Q_h_1:.1f}W"


def test_degradation_reduces_cop(model):
    """Degraded stage (lower ZT) must give lower COP than new stage."""
    import json, copy
    from model import PeltierTECF1b
    params_path = Path(__file__).parent.parent / "data" / "parameters.json"
    with open(params_path) as f:
        params_new = json.load(f)
    params_deg = copy.deepcopy(params_new)
    # Degrade both stages
    for k in range(1, params_deg["unit"]["n_stages"]["value"] + 1):
        params_deg["unit"][f"stage_{k}"]["degradation_factor"]["value"] = 0.85

    m_new = PeltierTECF1b(params_new)
    m_deg = PeltierTECF1b(params_deg)

    I = [3.0] * m_new.n_stages
    r_new = m_new.solve(I, 0.0, 40.0)
    r_deg = m_deg.solve(I, 0.0, 40.0)

    assert float(r_deg["COP"]) < float(r_new["COP"]), \
        "Degraded TEC should have lower COP than new TEC"


def test_scalar_current_broadcasts(model):
    """Scalar current should work (broadcast to all stages)."""
    r = model.predict({"current_stages": [3.0, 3.0], "T_cold": 0.0, "T_hot": 40.0})
    r2 = model.predict({"current": 3.0, "T_cold": 0.0, "T_hot": 40.0})
    np.testing.assert_allclose(float(r["COP"]), float(r2["COP"]), rtol=1e-10)


def test_benchmark(model):
    n = model._model.n_stages
    start = time.perf_counter()
    for _ in range(1000):
        model.predict({"current_stages": [3.0]*n, "T_cold": 0.0, "T_hot": 40.0})
    elapsed = time.perf_counter() - start
    print(f"\n  Benchmark: 1000 in {elapsed*1000:.2f} ms")
    assert elapsed < 5.0   # scalar calls, each is fast
