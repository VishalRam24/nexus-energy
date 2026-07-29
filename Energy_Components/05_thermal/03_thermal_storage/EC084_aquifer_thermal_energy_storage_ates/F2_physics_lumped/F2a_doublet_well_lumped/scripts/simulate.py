"""
EC084 -- ATES F2a Doublet-Well -- simulation report (optional Plotly).
Plotly import is guarded so its absence does not crash anything.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run(n_cycles=5):
    cm = ComponentModel()
    r = cm.predict({"n_cycles": n_cycles})
    print(f"Thermal radius: {r['thermal_radius_m']:.2f} m")
    print(f"Thermal capacitance: {r['thermal_capacitance_J_per_K']:.3e} J/K")
    print(f"Seasonal recovery eff: {[round(e, 3) for e in r['seasonal_efficiency']]}")
    print(f"Overall recovery eff: {r['recovery_efficiency']:.4f}")
    print(f"E injected: {r['E_injected_kWh']:.0f} kWh | "
          f"E extracted: {r['E_extracted_kWh']:.0f} kWh")
    return r


def make_report(r, out_html="simulation_report.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Mean bubble temperature",
                                        "Stored energy",
                                        "Charge(+1)/Discharge(-1) schedule"))
    fig.add_trace(go.Scatter(x=r["t_days"], y=r["T_storage"],
                             name="T_storage [°C]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t_days"], y=r["E_stored_kWh"],
                             name="E_stored [kWh]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t_days"], y=r["mode"],
                             name="mode", line_shape="hv"), row=3, col=1)
    fig.update_xaxes(title_text="time [days]", row=3, col=1)
    fig.update_layout(title=f"EC084 ATES F2a — recovery η={r['recovery_efficiency']:.3f}",
                      height=800)
    path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(path)
    print(f"[simulate] wrote {path}")
    return path


if __name__ == "__main__":
    res = run()
    make_report(res)
