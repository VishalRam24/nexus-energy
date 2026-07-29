"""
EC057 -- Stirling Dish CSP -- F2a Physics-Lumped
Simulation scenarios + optional Plotly HTML report.
Plotly import is wrapped so its absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def diurnal_dni(peak=950.0, sunrise=6.0, sunset=18.0):
    """Half-sine diurnal DNI profile [W/m2] as a function of time-of-day [s]."""
    def f(t):
        hod = (t / 3600.0) % 24.0
        if hod < sunrise or hod > sunset:
            return 0.0
        x = (hod - sunrise) / (sunset - sunrise)
        return peak * np.sin(np.pi * x)
    return f


def run_scenarios():
    m = ComponentModel()

    # Scenario A: cold-start warm-up at constant DNI
    warm = m.predict({"DNI": 900.0, "T_rec_init": 25.0, "T_amb": 25.0,
                      "dt": 10.0, "duration_s": 2400.0})

    # Scenario B: diurnal cycle (start 5:00, run 16 h)
    f = diurnal_dni()
    diurnal = m.predict({"DNI": f, "T_rec_init": 25.0, "T_amb": 20.0,
                         "dt": 120.0, "duration_s": 16 * 3600.0})
    # shift t to time-of-day starting 05:00
    t0 = 5.0 * 3600.0
    diurnal["t"] = diurnal["t"] + t0

    return warm, diurnal


def build_report(warm, diurnal, out_html):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -- skip gracefully
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Warm-up: Receiver Temperature", "Warm-up: Net Power",
                        "Diurnal: Power vs Time-of-day", "Warm-up: Efficiencies"),
    )
    fig.add_trace(go.Scatter(x=warm["t"] / 60.0, y=warm["T_rec_c"],
                             name="T_rec [C]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=warm["t"] / 60.0, y=warm["P_elec_kw"],
                             name="P_elec [kW]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=diurnal["t"] / 3600.0, y=diurnal["P_elec_kw"],
                             name="P_elec diurnal [kW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=warm["t"] / 60.0, y=warm["eta_carnot"],
                             name="Carnot"), row=2, col=2)
    fig.add_trace(go.Scatter(x=warm["t"] / 60.0, y=warm["eta_stirling"],
                             name="Stirling"), row=2, col=2)
    fig.update_layout(title="EC057 Stirling Dish CSP -- F2a Receiver ODE + Engine",
                      height=720, width=1100)
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    warm, diurnal = run_scenarios()
    print(f"Warm-up final: T_rec={warm['T_rec_c'][-1]:.1f} C, "
          f"P={warm['P_elec_kw'][-1]:.2f} kW, eta_sys={warm['eta_system'][-1]*100:.1f}%")
    peak_kwh = np.trapz(diurnal["P_elec_kw"], diurnal["t"]) / 3600.0
    print(f"Diurnal energy: {peak_kwh:.1f} kWh, peak {np.max(diurnal['P_elec_kw']):.1f} kW")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    build_report(warm, diurnal, out)
