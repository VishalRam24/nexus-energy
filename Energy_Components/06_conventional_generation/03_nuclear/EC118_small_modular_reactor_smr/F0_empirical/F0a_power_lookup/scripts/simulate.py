"""Optional Plotly report: electrical power and efficiency vs load factor."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    load = np.linspace(m.lookup.load_bp[0], 1.0, 50)
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not available; power table:")
        for L in m.lookup.load_bp:
            print(f"  load={L:.2f}  P_e={m.lookup.power_elec(L):.1f} MW  eta={m.lookup.efficiency(L):.4f}")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=load, y=m.lookup.power_elec(load), mode="lines", name="P_elec [MW]"))
    fig.add_trace(go.Scatter(x=load, y=m.lookup.efficiency(load) * m.lookup.pelec_bp[-1],
                             mode="lines", name="eta (scaled)", line=dict(dash="dot")))
    fig.update_layout(title=f"{m.component_id} {m.component_name} - F0a power lookup",
                      xaxis_title="Load factor", yaxis_title="Electrical power [MW]")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
