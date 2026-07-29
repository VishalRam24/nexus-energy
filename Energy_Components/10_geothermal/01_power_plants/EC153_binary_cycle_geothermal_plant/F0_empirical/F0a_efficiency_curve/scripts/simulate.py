"""Optional Plotly report for EC153 F0a efficiency curve."""
import os

import numpy as np

from predict import ComponentModel


def main():
    m = ComponentModel()
    T = np.linspace(80, 200, 50)
    eta = [m.predict({"T_geothermal": t, "flow_rate_kgs": 50})["eta_net"] for t in T]
    p = [m.predict({"T_geothermal": t, "flow_rate_kgs": 50})["net_power_kW"] for t in T]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly optional
        print("plotly unavailable, skipping report:", e)
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Net efficiency", "Net power (50 kg/s)"))
    fig.add_trace(go.Scatter(x=T, y=eta, name="eta_net"), 1, 1)
    fig.add_trace(go.Scatter(x=T, y=p, name="P_net"), 1, 2)
    fig.update_layout(title="EC153 Binary Cycle Geothermal Plant — F0a empirical curve")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
