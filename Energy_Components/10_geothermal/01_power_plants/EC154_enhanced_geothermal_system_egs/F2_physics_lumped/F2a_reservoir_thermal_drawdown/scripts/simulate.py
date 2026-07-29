"""
EC154 -- EGS F2a -- optional Plotly simulation report.
Plotly import is guarded so absence does not crash.
Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()
    r = cm.predict({"years": 40.0, "n_points": 300})

    t = r["t_years"]
    print("EGS 40-year thermal-drawdown summary")
    print(f"  tau_res = {r['tau_res_yr']:.1f} yr, HX eff = {r['effectiveness']:.3f}")
    for yr in [0, 5, 10, 20, 30, 40]:
        i = min(range(len(t)), key=lambda k: abs(t[k] - yr))
        print(f"  yr {t[i]:5.1f}: T_rock={r['T_rock_degC'][i]:6.1f}C  "
              f"T_prod={r['T_prod_degC'][i]:6.1f}C  P_net={r['P_net_kW'][i]:7.0f} kW  "
              f"eta={r['eta_cycle'][i]:.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Reservoir & produced temperature", "Net electrical power",
        "Cycle vs Carnot efficiency", "Heat input to cycle"))
    fig.add_trace(go.Scatter(x=t, y=r["T_rock_degC"], name="T_rock"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["T_prod_degC"], name="T_prod"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["P_net_kW"], name="P_net"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=r["P_gross_kW"], name="P_gross"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=r["eta_carnot"], name="eta_carnot"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["eta_cycle"], name="eta_cycle"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["Q_in_kW"], name="Q_in"), row=2, col=2)
    fig.update_layout(title="EC154 EGS F2a -- Reservoir Thermal Drawdown",
                      height=800, template="plotly_white")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
