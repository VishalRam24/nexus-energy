"""EC220 TENG F0a - optional Plotly power report."""
import os
import numpy as np
from model import TENGPowerLookup


def main():
    c = TENGPowerLookup()
    fs = np.linspace(0.1, 100, 60)
    ps = [c.power_mW(f) for f in fs]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fs, y=ps, mode="lines", name="power"))
        fig.update_layout(title="EC220 TENG F0a average power vs cycle frequency",
                          xaxis_title="frequency (Hz)", yaxis_title="power (mW)")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for f, v in zip(fs[::10], ps[::10]):
            print(round(float(f), 2), round(v, 6))


if __name__ == "__main__":
    main()
