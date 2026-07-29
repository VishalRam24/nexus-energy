"""
EC217 -- TEC F2a -- Simulation scenarios + optional Plotly HTML report.
Plotly is imported lazily so its absence never crashes the script.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1) Transient pull-down at a few currents
    currents = [1.0, 3.0, 5.0, 8.0]
    pulldowns = {I: m.simulate(I=I, Q_load=1.0, duration_s=200.0) for I in currents}

    # 2) Steady Q_cold and COP vs current (algebraic, at fixed plates)
    T_c, T_h = 280.0, 305.0
    I_sweep = np.linspace(0.0, m.current_for_max_cooling(T_c, T_h) * 1.6, 80)
    Qc = np.array([m.junction_heat(T_c, T_h, I)["Q_cold_W"] for I in I_sweep])
    COP = np.array([m.junction_heat(T_c, T_h, I)["COP"] for I in I_sweep])

    print("=== EC217 TEC F2a simulation ===")
    for I, out in pulldowns.items():
        ss = out["steady_state"]
        print(f" I={I:4.1f} A -> T_cold_ss={ss['T_cold_ss_K']:6.2f} K, "
              f"dT={ss['dT_ss_K']:5.2f} K, COP={ss['COP']:.3f}, ZT={ss['ZT_avg']:.3f}")
    print(f" Q_cold,max={Qc.max():.2f} W at I={I_sweep[np.argmax(Qc)]:.2f} A; "
          f"I_qmax={m.current_for_max_cooling(T_c, T_h):.2f} A")
    print(f" max temperature lift (zero load) = {m.max_temperature_lift(T_h):.2f} K")
    return pulldowns, (I_sweep, Qc, COP, currents)


def build_report(path=None):
    pulldowns, (I_sweep, Qc, COP, currents) = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("plotly not available -- skipping HTML report.")
        return

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Cold-plate pull-down", "Q_cold & COP vs current"))
    for I in currents:
        out = pulldowns[I]
        fig.add_trace(go.Scatter(x=out["t"], y=out["T_cold"], name=f"I={I} A"), row=1, col=1)
    fig.add_trace(go.Scatter(x=I_sweep, y=Qc, name="Q_cold [W]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=I_sweep, y=COP, name="COP", yaxis="y3"), row=1, col=2)
    fig.update_xaxes(title_text="t [s]", row=1, col=1)
    fig.update_yaxes(title_text="T_cold [K]", row=1, col=1)
    fig.update_xaxes(title_text="I [A]", row=1, col=2)
    fig.update_layout(title="EC217 TEC F2a -- Peltier + Lumped Thermal ODE")
    fig.write_html(path)
    print(f"Report written: {path}")


if __name__ == "__main__":
    build_report()
