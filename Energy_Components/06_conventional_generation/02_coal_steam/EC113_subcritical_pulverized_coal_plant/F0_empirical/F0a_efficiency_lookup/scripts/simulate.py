"""Optional Plotly report: efficiency vs part-load for the F0a lookup."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plr = np.linspace(m.lookup.plr_bp[0], 1.0, 50)
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not available; efficiency table:")
        for p in m.lookup.plr_bp:
            print(f"  PLR={p:.2f}  eta={m.lookup.efficiency(p):.4f}")
        return
    fig = go.Figure()
    for T in (0.0, 15.0, 35.0):
        fig.add_trace(go.Scatter(x=plr, y=m.lookup.efficiency(plr, T),
                                 mode="lines", name=f"T_amb={T:.0f} C"))
    fig.update_layout(title=f"{m.component_id} {m.component_name} - F0a efficiency lookup",
                      xaxis_title="Part-load ratio", yaxis_title="Net LHV efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
