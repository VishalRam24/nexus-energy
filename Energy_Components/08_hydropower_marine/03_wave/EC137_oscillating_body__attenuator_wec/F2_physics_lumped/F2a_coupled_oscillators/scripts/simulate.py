"""
EC137 -- Attenuator WEC -- F2a Coupled Oscillators
Simulation scenarios + interactive Plotly report (optional).

Run:  python3 scripts/simulate.py
Plotly is optional; if absent, prints a text summary instead.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run():
    cm = ComponentModel()
    m = cm._model

    # --- Scenario 1: time-domain response in a moderate sea state ----------
    Hs, Te = 3.0, 9.0
    r = cm.predict({"H_s": Hs, "T_e": Te, "dt": 0.05, "duration_s": 120.0})

    # --- Scenario 2: PTO damping sweep -> optimal B_pto -------------------
    opt = m.optimal_B_pto(Hs, Te, dt=0.1, duration_s=80.0, n_scan=15)

    # --- Scenario 3: power matrix over sea states ------------------------
    Hs_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Te_grid = np.array([7.0, 8.0, 9.0, 10.0, 12.0])
    Pmat = np.zeros((len(Hs_grid), len(Te_grid)))
    for i, h in enumerate(Hs_grid):
        for j, t in enumerate(Te_grid):
            Pmat[i, j] = m.simulate(h, t, B_pto=1.3e8, dt=0.15,
                                    duration_s=60.0)["mean_power_elec_W"] / 1e3

    print(f"Scenario 1 (Hs={Hs} m, Te={Te} s):")
    print(f"  Mean electrical power : {r['mean_power_elec_W']/1e3:.1f} kW")
    print(f"  Capture width         : {r['capture_width_m']:.2f} m  (CWR {r['capture_width_ratio']:.2f})")
    print(f"  Energy residual       : {r['energy_residual']*100:.2f} %")
    print(f"Scenario 2 optimal B_pto: {opt['B_opt']:.2e} N.m.s/rad -> {opt['P_max_elec_W']/1e3:.1f} kW")
    print("Scenario 3 power matrix [kW]:")
    print("  Te:    " + "  ".join(f"{t:6.1f}" for t in Te_grid))
    for i, h in enumerate(Hs_grid):
        print(f"  Hs {h:.0f}: " + "  ".join(f"{Pmat[i,j]:6.1f}" for j in range(len(Te_grid))))

    if not _HAVE_PLOTLY:
        print("\n[plotly not installed -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Relative hinge angles theta_i(t)",
                        "Instantaneous electrical power",
                        "PTO damping sweep (optimal B_pto)",
                        "Power matrix (kW)"),
        specs=[[{}, {}], [{}, {"type": "heatmap"}]],
    )
    for i in range(m.n_joint):
        fig.add_trace(go.Scatter(x=r["t"], y=np.degrees(r["theta"][i]),
                                 name=f"joint {i+1}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["power_total_elec"] / 1e3,
                             name="P_elec", line=dict(color="green")), row=1, col=2)
    fig.add_trace(go.Scatter(x=opt["B_grid"], y=opt["P_grid"] / 1e3,
                             mode="lines+markers", name="P vs B_pto"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[opt["B_opt"]], y=[opt["P_max_elec_W"] / 1e3],
                             mode="markers", marker=dict(size=12, color="red"),
                             name="optimum"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=Pmat, x=[f"{t:.0f}s" for t in Te_grid],
                             y=[f"{h:.0f}m" for h in Hs_grid],
                             colorscale="Viridis", name="kW"), row=2, col=2)
    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_yaxes(title_text="angle [deg]", row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=1, col=2)
    fig.update_yaxes(title_text="P [kW]", row=1, col=2)
    fig.update_xaxes(title_text="B_pto [N.m.s/rad]", type="log", row=2, col=1)
    fig.update_yaxes(title_text="P [kW]", row=2, col=1)
    fig.update_layout(title="EC137 Attenuator WEC -- F2a Coupled Oscillators",
                      height=800, showlegend=True)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"\nReport written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
