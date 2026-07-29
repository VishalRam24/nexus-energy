"""
EC221 -- MHD Generator -- F2a Physics-Lumped Channel
Simulation scenarios + optional Plotly HTML report.
Plotly import is wrapped so absence does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # 1) Axial channel profiles at the max-power load factor
    base = cm.predict({"K_load": 0.5})

    # 2) Load-factor sweep: power and electrical efficiency
    Ks = np.linspace(0.05, 0.95, 19)
    P_K = np.array([cm.predict({"K_load": float(k)})["P_elec_W"] for k in Ks]) / 1e6
    eta_K = np.array([cm.predict({"K_load": float(k)})["eta_electric"] for k in Ks])

    # 3) Magnetic-field sweep at K=0.5
    Bs = np.linspace(1.0, 8.0, 15)
    P_B = np.array([cm.predict({"B_field_T": float(b), "K_load": 0.5})["P_elec_W"]
                    for b in Bs]) / 1e6

    # 4) Inlet-temperature sweep (Saha conductivity sensitivity)
    Ts = np.linspace(2200.0, 3400.0, 13)
    P_T = np.array([cm.predict({"T_inlet": float(t), "K_load": 0.5})["P_elec_W"]
                    for t in Ts]) / 1e6

    return base, (Ks, P_K, eta_K), (Bs, P_B), (Ts, P_T)


def main():
    base, (Ks, P_K, eta_K), (Bs, P_B), (Ts, P_T) = run_scenarios()
    print("=== EC221 MHD Generator F2a -- channel summary (K=0.5) ===")
    print(f"  P_elec        = {base['P_elec_W']/1e6:.3f} MW")
    print(f"  eta_electric  = {base['eta_electric']:.3f}")
    print(f"  enth_extract  = {base['eta_enthalpy_extraction']:.4f}")
    print(f"  beta_hall     = {base['beta_hall']:.2f}")
    print(f"  u  : {base['u'][0]:.0f} -> {base['u'][-1]:.0f} m/s")
    print(f"  T  : {base['T'][0]:.0f} -> {base['T'][-1]:.0f} K")
    print(f"  p  : {base['p'][0]/1e3:.1f} -> {base['p'][-1]/1e3:.1f} kPa")
    print(f"  sigma : {base['sigma'][0]:.2f} -> {base['sigma'][-1]:.2f} S/m")
    print(f"  K at max power = {Ks[int(np.argmax(P_K))]:.2f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:
        print(f"[plotly unavailable: {exc}] skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Axial profiles (K=0.5)", "Power & eta vs load factor K",
                        "Power vs magnetic field B", "Power vs inlet temperature"),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{}, {}]],
    )
    x = base["x"]
    fig.add_trace(go.Scatter(x=x, y=base["u"], name="u [m/s]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=base["T"], name="T [K]"),
                  row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=Ks, y=P_K, name="P_elec [MW]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=Ks, y=eta_K, name="eta_electric"),
                  row=1, col=2, secondary_y=True)

    fig.add_trace(go.Scatter(x=Bs, y=P_B, name="P vs B [MW]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=Ts, y=P_T, name="P vs T_in [MW]"), row=2, col=2)

    fig.update_layout(title="EC221 MHD Generator -- F2a Physics-Lumped Channel",
                      height=820, width=1100)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {out}")


if __name__ == "__main__":
    main()
