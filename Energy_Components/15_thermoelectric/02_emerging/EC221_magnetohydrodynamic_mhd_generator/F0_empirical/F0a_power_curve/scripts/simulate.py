"""EC221 MHD Generator F0a - optional Plotly power report."""
import os
import numpy as np
from model import MHDPowerCurve


def main():
    c = MHDPowerCurve()
    us = np.linspace(200, 2000, 60)
    pds = [c.power_density(u) for u in us]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=us, y=pds, mode="lines", name="power density"))
        fig.update_layout(title="EC221 MHD F0a power density vs plasma velocity",
                          xaxis_title="velocity (m/s)", yaxis_title="W/m^3")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for u, v in zip(us[::10], pds[::10]):
            print(round(float(u), 1), round(v, 1))


if __name__ == "__main__":
    main()
