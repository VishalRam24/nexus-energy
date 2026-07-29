"""EC218 Thermionic Converter F0a - optional Plotly power curve report."""
import os
import numpy as np
from model import ThermionicPowerCurve


def main():
    c = ThermionicPowerCurve()
    ts = np.linspace(1200, 2000, 60)
    pds = [c.power_density(t) for t in ts]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=pds, mode="lines", name="power density"))
        fig.update_layout(title="EC218 Thermionic F0a power density vs emitter temperature",
                          xaxis_title="T_emitter (K)", yaxis_title="W/m^2", yaxis_type="log")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for t, v in zip(ts[::10], pds[::10]):
            print(round(float(t), 1), round(v, 1))


if __name__ == "__main__":
    main()
