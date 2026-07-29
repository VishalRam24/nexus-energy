"""
EC202 -- DAC Liquid Solvent -- F2a Contactor + Calciner
Plotly HTML simulation report generator (optional; safe if plotly absent).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    model = ComponentModel()

    # Scenario 1: cold-start calciner, 30-day startup transient
    r1 = model.predict({"dt": 3600.0, "duration_s": 3600.0 * 24 * 30,
                        "T_calciner_K": 1000.0})

    # Scenario 2: sweep air velocity -> single-pass capture vs throughput
    u_vals = np.linspace(0.5, 4.0, 25)
    eta = np.array([model._model.single_pass_capture(u_air=u) for u in u_vals])
    R = np.array([model._model.absorption_rate(u_air=u) for u in u_vals])

    if not _HAVE_PLOTLY:
        print("plotly not installed; skipping HTML report.")
        print(f"Scenario 1 final: T={r1['T_calciner_final_C']:.1f} C, "
              f"SEC={r1['sec_thermal_final_GJ_tCO2']:.2f} GJ/tCO2")
        return

    days = r1["t"] / 86400.0
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Calciner temperature (cold start)",
        "Specific thermal energy (GJ/tCO2)",
        "Single-pass capture vs air velocity",
        "Total CO2 absorption rate vs air velocity"))

    fig.add_trace(go.Scatter(x=days, y=r1["T_calciner_K"] - 273.15,
                             name="T_calciner [C]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=r1["sec_thermal_GJ_tCO2"],
                             name="SEC_thermal"), row=1, col=2)
    fig.add_trace(go.Scatter(x=u_vals, y=eta, name="eta_single"), row=2, col=1)
    fig.add_trace(go.Scatter(x=u_vals, y=R, name="R_abs [mol/s]"), row=2, col=2)

    fig.update_layout(title="EC202 DAC Liquid Solvent — F2a Contactor + Calciner",
                      height=800, showlegend=False)
    out = os.path.join(OUTPUT_DIR, "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
