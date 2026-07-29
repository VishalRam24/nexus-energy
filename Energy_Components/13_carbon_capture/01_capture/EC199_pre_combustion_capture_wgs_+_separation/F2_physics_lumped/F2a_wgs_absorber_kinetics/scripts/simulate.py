"""
EC199 -- Pre-Combustion Capture (WGS + Separation) -- F2a Physics-Lumped
Simulation scenarios + optional interactive Plotly report.

Scenarios:
  (1) WGS reactor extent vs residence time at the LT-shift optimum.
  (2) Absorber CO2 uptake vs contact time.
  (3) Capture rate & H2 purity vs operating pressure (high-P advantage).
  (4) WGS conversion vs temperature (LT-shift window) against equilibrium.

Run: python3 scripts/simulate.py
Plotly is optional -- absence does not crash the script.
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


def run_scenarios():
    cm = ComponentModel()

    # Base case
    base = cm.predict({"syngas_flow_mol_s": 1000.0, "co_fraction": 0.45,
                       "h2_fraction": 0.35, "P_bar": 30.0})

    # Pressure sweep
    P_vals = np.linspace(12.0, 55.0, 15)
    cr_P, hp_P, pco2_P = [], [], []
    for P in P_vals:
        r = cm.predict({"P_bar": float(P)})
        cr_P.append(r["capture_rate"])
        hp_P.append(r["h2_purity"])
        pco2_P.append(r["p_CO2_absorber_in_bar"])

    # Temperature sweep
    T_vals = np.linspace(463.15, 673.15, 20)
    X_T, Xeq_T = [], []
    for T in T_vals:
        r = cm.predict({"T_WGS_K": float(T), "P_bar": 30.0})
        X_T.append(r["wgs_conversion"])
        Xeq_T.append(r["wgs_equilibrium_conversion"])

    return {
        "base": base,
        "P_vals": P_vals, "cr_P": cr_P, "hp_P": hp_P, "pco2_P": pco2_P,
        "T_vals": T_vals, "X_T": X_T, "Xeq_T": Xeq_T,
    }


def make_report(data, out_html=None):
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    if not _HAVE_PLOTLY:
        print("[simulate] Plotly not available -- skipping HTML report.")
        return None

    b = data["base"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "WGS reactor: shift extent vs residence time",
            "Absorber: CO2 uptake vs contact time",
            "Capture rate & H2 purity vs pressure",
            "WGS conversion vs temperature (LT-shift)",
        ),
    )

    wgs = b["wgs"]
    fig.add_trace(go.Scatter(x=wgs["t"], y=wgs["xi_t"], name="shift extent [mol/s]"),
                  row=1, col=1)

    ab = b["absorber"]
    fig.add_trace(go.Scatter(x=ab["t"], y=ab["n_abs_t"], name="CO2 absorbed [mol/s]"),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=data["P_vals"], y=np.array(data["cr_P"]) * 100,
                             name="capture rate [%]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data["P_vals"], y=np.array(data["hp_P"]) * 100,
                             name="H2 purity [%]"), row=2, col=1)

    fig.add_trace(go.Scatter(x=data["T_vals"] - 273.15, y=np.array(data["X_T"]) * 100,
                             name="X_CO (kinetic) [%]"), row=2, col=2)
    fig.add_trace(go.Scatter(x=data["T_vals"] - 273.15, y=np.array(data["Xeq_T"]) * 100,
                             name="X_CO equilibrium [%]", line=dict(dash="dash")),
                  row=2, col=2)

    fig.update_layout(title="EC199 Pre-Combustion Capture (WGS + Selexol) -- F2a",
                      height=800, showlegend=True)
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    data = run_scenarios()
    b = data["base"]
    print("EC199 Pre-Combustion Capture F2a -- base case (30 bar, S/C=3, 250 C):")
    print(f"  WGS conversion   : {b['wgs_conversion']*100:.1f} % "
          f"(eq {b['wgs_equilibrium_conversion']*100:.1f} %)")
    print(f"  Capture rate     : {b['capture_rate']*100:.1f} %")
    print(f"  H2-rich fuel     : {b['h2_rich_fuel_mol_s']:.1f} mol/s "
          f"(purity {b['h2_purity']*100:.1f} %)")
    print(f"  Energy penalty   : {b['energy_penalty_GJ_tCO2']:.2f} GJ/tCO2 "
          f"({b['power_penalty_MW']:.1f} MW)")
    print(f"  WGS heat         : {b['wgs_heat_kW']:.0f} kW")
    make_report(data)
