"""
EC088 -- Oil-Fired Boiler -- F2a Dynamic Thermal Mass
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash; run with `python3 scripts/simulate.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()

    # Cold-start to setpoint with a mid-run firing-rate step.
    def fire(t):
        return 0.4 if t < 1200 else 0.95

    r = cm.predict({
        "firing_rate": fire,
        "T_water_init": 25.0,
        "T_return": 55.0,
        "T_ambient": 15.0,
        "dt": 5.0,
        "duration_s": 3600.0,
    })

    t_min = r["t"] / 60.0
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); printing summary instead.")
        print(f"  Final T_water   : {r['T_water_C'][-1]:.2f} C")
        print(f"  Final eta_comb  : {r['eta_combustion'][-1]:.4f}")
        print(f"  Final eta_overall: {r['eta_overall'][-1]:.4f}")
        print(f"  Final Q_fuel    : {r['Q_fuel_W'][-1]/1000:.2f} kW")
        return None

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Boiler water temperature", "Heat flows", "Efficiencies"),
    )
    fig.add_trace(go.Scatter(x=t_min, y=r["T_water_C"], name="T_water"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["T_flue_C"], name="T_flue"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["Q_fuel_W"]/1000, name="Q_fuel kW"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["Q_useful_W"]/1000, name="Q_useful kW"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["Q_sensible_loss_W"]/1000, name="stack sensible kW"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["Q_latent_loss_W"]/1000, name="stack latent kW"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["eta_combustion"], name="eta_comb"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_min, y=r["eta_overall"], name="eta_overall"), row=3, col=1)
    fig.update_xaxes(title_text="time [min]", row=3, col=1)
    fig.update_layout(title="EC088 Oil-Fired Boiler -- F2a Dynamic Thermal Mass",
                      height=900)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] report written to {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
