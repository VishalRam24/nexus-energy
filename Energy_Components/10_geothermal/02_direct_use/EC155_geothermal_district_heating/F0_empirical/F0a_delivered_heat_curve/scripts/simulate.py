"""Optional Plotly report for EC155 F0a delivered-heat curve."""
import os

import numpy as np

from predict import ComponentModel


def main():
    m = ComponentModel()
    T = np.linspace(50, 150, 50)
    q = [m.predict({"T_source": t, "flow_rate_kgs": 50})["q_specific_kW_per_kgps"] for t in T]
    Q = [m.predict({"T_source": t, "flow_rate_kgs": 50})["Q_delivered_kW"] for t in T]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print("plotly unavailable, skipping report:", e)
        return
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Delivered heat per kg/s", "Delivered heat (50 kg/s)"))
    fig.add_trace(go.Scatter(x=T, y=q, name="q_specific"), 1, 1)
    fig.add_trace(go.Scatter(x=T, y=Q, name="Q_delivered"), 1, 2)
    fig.update_layout(title="EC155 Geothermal District Heating — F0a delivered-heat curve")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
