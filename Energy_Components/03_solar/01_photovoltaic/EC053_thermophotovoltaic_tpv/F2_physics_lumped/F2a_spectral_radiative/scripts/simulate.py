"""
EC053 -- Thermophotovoltaic (TPV) -- F2a Spectral Radiative
Simulation scenarios + optional Plotly HTML report.

Run: python3 scripts/simulate.py
Plotly import is guarded so absence does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def scenario_temperature_sweep(model):
    """Steady MPP power and efficiency vs emitter temperature."""
    Ts = np.linspace(900.0, 2000.0, 40)
    P, eta_sys, eta_spec, Voc = [], [], [], []
    for T in Ts:
        eff = model._model.efficiencies(T, 300.0)
        P.append(eff["P_elec_W"] * 1e3)        # mW
        eta_sys.append(eff["eta_system"] * 100)
        eta_spec.append(eff["eta_spectral"] * 100)
        Voc.append(eff["Voc"])
    return Ts, np.array(P), np.array(eta_sys), np.array(eta_spec), np.array(Voc)


def scenario_iv_curve(model, T_emitter=1500.0, T_cell=300.0):
    m = model._model
    Jph = m.photocurrent_density(T_emitter)
    Voc = m.open_circuit_voltage(T_emitter, T_cell, Jph=Jph)
    V = np.linspace(0.0, Voc, 100)
    J = np.array([m.current_density(v, T_emitter, T_cell, Jph=Jph) for v in V])
    return V, J, J * V


def scenario_thermal_transient(model):
    return model.predict({"T_emitter_K": 1600.0, "T_cell0_K": 300.0,
                          "dt": 0.5, "duration_s": 120.0})


def main():
    model = ComponentModel()
    Ts, P, eta_sys, eta_spec, Voc = scenario_temperature_sweep(model)
    V, J, Pden = scenario_iv_curve(model)
    tr = scenario_thermal_transient(model)

    print("=== EC053 TPV F2a simulation ===")
    print(f"Peak elec power over 900-2000 K: {P.max():.2f} mW @ {Ts[np.argmax(P)]:.0f} K")
    print(f"Peak system efficiency: {eta_sys.max():.1f} %")
    print(f"I-V @1500K: Voc={V[-1]:.3f} V, Jsc={J[0]:.1f} A/m^2, Pmax_den={Pden.max():.1f} W/m^2")
    print(f"Thermal transient @1600K: T_cell 300 -> {tr['T_cell'][-1]:.1f} K")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly not available: {e}] -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Electrical power vs emitter T",
                        "Efficiencies vs emitter T",
                        "I-V & P-V curve @1500K",
                        "Cell thermal transient @1600K"),
    )
    fig.add_trace(go.Scatter(x=Ts, y=P, name="P_elec [mW]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=eta_sys, name="eta_system [%]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=Ts, y=eta_spec, name="eta_spectral [%]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=V, y=J, name="J [A/m^2]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=V, y=Pden, name="P density [W/m^2]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["T_cell"], name="T_cell [K]"), row=2, col=2)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
