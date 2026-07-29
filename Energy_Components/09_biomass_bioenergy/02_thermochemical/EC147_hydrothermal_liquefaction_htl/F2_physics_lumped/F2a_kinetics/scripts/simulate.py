"""
EC147 -- Hydrothermal Liquefaction (HTL) -- F2a Physics-Lumped Kinetics
Optional Plotly report: composition vs time, biocrude-yield severity surface.
Plotly import is guarded so its absence does not crash anything.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT_HTML = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def build_report():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly not available ({e}); skipping HTML report.")
        return None

    cm = ComponentModel()

    # (1) Composition trajectory at 350 C, 30 min, starting hot (300 C).
    r = cm.predict({"T_setpoint_C": 350.0, "residence_min": 60.0, "T0_C": 300.0})

    # (2) Biocrude-yield severity sweep (T x residence time).
    temps = np.linspace(280.0, 370.0, 19)
    times = np.linspace(2.0, 90.0, 19)
    Z = np.zeros((len(temps), len(times)))
    for i, T in enumerate(temps):
        for j, tt in enumerate(times):
            Z[i, j] = cm._model.biocrude_yield_at(T, tt, T0_C=T)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Product lumps vs time (350 C)",
                        "Biocrude yield surface (severity)"),
        specs=[[{"type": "xy"}, {"type": "heatmap"}]],
    )
    for name in ["biomass", "biocrude", "aqueous", "gas", "solid"]:
        fig.add_trace(
            go.Scatter(x=r["t_min"], y=r[name], mode="lines", name=name),
            row=1, col=1,
        )
    fig.add_trace(
        go.Heatmap(x=times, y=temps, z=Z, colorscale="Viridis",
                   colorbar=dict(title="biocrude")),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="time [min]", row=1, col=1)
    fig.update_yaxes(title_text="mass fraction", row=1, col=1)
    fig.update_xaxes(title_text="residence time [min]", row=1, col=2)
    fig.update_yaxes(title_text="temperature [C]", row=1, col=2)
    fig.update_layout(title="EC147 HTL F2a — lumped kinetics + energy balance")

    fig.write_html(_OUT_HTML)
    print(f"[simulate] wrote {_OUT_HTML}")
    return _OUT_HTML


if __name__ == "__main__":
    build_report()
