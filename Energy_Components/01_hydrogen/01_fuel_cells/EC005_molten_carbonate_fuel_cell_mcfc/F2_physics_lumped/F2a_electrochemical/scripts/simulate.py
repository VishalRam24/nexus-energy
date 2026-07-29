"""
EC005 -- MCFC -- F2a Electrochemical: simulation scenarios + optional Plotly report.

Generates a polarization curve and a cold-start thermal transient and (if Plotly
is available) writes an interactive HTML report. Plotly import is wrapped so its
absence does not crash the module.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

P = dict(pH2=0.7, pO2=0.15, pH2O=0.20, pCO2_cathode=0.15, pCO2_anode=0.10)


def polarization_curve(cm, T=923.15):
    m = cm._model
    j = np.linspace(0.005, 0.58, 60)
    V = np.array([m.cell_voltage(jj, T, P["pH2"], P["pO2"], P["pH2O"],
                                 P["pCO2_cathode"], P["pCO2_anode"]) for jj in j])
    return j, V, j * V


def thermal_transient(cm):
    r = cm.predict({"current_density_A_cm2": 0.35, "T_cell_K": 873.15,
                    "dt": 5.0, "duration_s": 1800.0, **P})
    return r


def main():
    cm = ComponentModel()
    j, V, Pden = polarization_curve(cm)
    tr = thermal_transient(cm)
    print(f"Polarization: V(0.3 A/cm2) ~ {np.interp(0.3, j, V):.4f} V, "
          f"peak power density {Pden.max():.4f} W/cm2")
    print(f"Cold-start: T 873.15 K -> {tr['temperature'][-1]:.2f} K over 1800 s")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Polarization V-j", "Power density", "Cold-start thermal transient"))
    fig.add_trace(go.Scatter(x=j, y=V, name="V_cell"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j, y=Pden, name="P density"), row=1, col=2)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["temperature"], name="T"), row=1, col=3)
    fig.update_layout(title="EC005 MCFC F2a — Electrochemical + Thermal ODE", showlegend=False)
    fig.write_html(_OUT)
    print(f"[simulate] Wrote {_OUT}")


if __name__ == "__main__":
    main()
