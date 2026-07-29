"""EC217 TEC F0a - optional Plotly COP map report."""
import os
import numpy as np
from model import TECCopMap


def main():
    c = TECCopMap()
    dts = np.linspace(5, 67, 60)
    cops = [c.cop(d) for d in dts]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dts, y=cops, mode="lines", name="COP"))
        fig.update_layout(title="EC217 TEC F0a COP vs temperature lift",
                          xaxis_title="deltaT (K)", yaxis_title="COP")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for d, v in zip(dts[::10], cops[::10]):
            print(round(float(d), 1), round(v, 4))


if __name__ == "__main__":
    main()
