"""Optional Plotly report for EC156 F0a COP map."""
import os

import numpy as np

from predict import ComponentModel


def main():
    m = ComponentModel()
    sinks = [35, 45, 55, 65]
    T_src = np.linspace(0, 25, 30)
    try:
        import plotly.graph_objects as go
    except Exception as e:
        print("plotly unavailable, skipping report:", e)
        return
    fig = go.Figure()
    for tk in sinks:
        cop = [m.predict({"T_source": t, "T_sink": tk})["COP"] for t in T_src]
        fig.add_trace(go.Scatter(x=T_src, y=cop, name=f"T_sink={tk}C"))
    fig.update_layout(title="EC156 Geothermal Heat Pump — F0a heating COP map",
                      xaxis_title="Ground-loop source temperature (C)", yaxis_title="COP")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
