"""
EC171 -- Cycloconverter -- F2a Physics-Lumped
Optional Plotly simulation report. Plotly import is wrapped so absence does not
crash; falls back to a text summary.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    r = cm.predict({"r_mod": 0.85, "f_out": 10.0, "n_cycles": 5})

    print("EC171 Cycloconverter F2a -- simulation summary")
    print(f"  f_out / f_line          : {r['f_out']} / {r['f_line']} Hz "
          f"(ratio {r['freq_ratio']:.3f}, <1/3: {r['below_one_third']})")
    print(f"  V_out_ll_rms            : {r['V_out_ll_rms']:.1f} V")
    print(f"  I_out_rms               : {r['I_out_rms']:.1f} A")
    print(f"  P_out / P_in            : {r['P_out_total']/1e3:.1f} / {r['P_in_total']/1e3:.1f} kW")
    print(f"  efficiency              : {r['efficiency']:.4f}")
    print(f"  input DPF (lagging)     : {r['input_displacement_factor']:.3f}")
    print(f"  output THD              : {r['output_thd']:.3f}")
    print(f"  dominant harmonics [Hz] : {r['dominant_harmonics_hz']}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("\n[plotly not available -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Averaged output voltage v_out(t)",
                        "Output current i_out(t) (R-L load ODE)",
                        "Firing angle alpha(t)"),
    )
    fig.add_trace(go.Scatter(x=r["t"], y=r["v_out_avg"], name="v_out_avg [V]"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["i_out"], name="i_out [A]"), 2, 1)
    fig.add_trace(go.Scatter(x=r["t"], y=np.degrees(r["alpha"]), name="alpha [deg]"), 3, 1)
    fig.update_layout(height=800, title_text="EC171 Cycloconverter F2a -- Phase-Controlled Averaged Model")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"\nWrote report: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
