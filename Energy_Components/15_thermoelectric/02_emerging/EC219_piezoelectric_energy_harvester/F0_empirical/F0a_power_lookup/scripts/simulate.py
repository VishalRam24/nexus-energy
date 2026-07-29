"""EC219 Piezoelectric Energy Harvester F0a - optional Plotly power report."""
import os
import numpy as np
from model import PiezoPowerLookup


def main():
    c = PiezoPowerLookup()
    accs = np.linspace(0.5, 50, 60)
    ps = [c.power_mW(a) for a in accs]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=accs, y=ps, mode="lines", name="power"))
        fig.update_layout(title="EC219 Piezo F0a harvested power vs base acceleration",
                          xaxis_title="acceleration (m/s^2)", yaxis_title="power (mW)")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for a, v in zip(accs[::10], ps[::10]):
            print(round(float(a), 2), round(v, 4))


if __name__ == "__main__":
    main()
