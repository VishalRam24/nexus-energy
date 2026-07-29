"""EC137 F0a — optional Plotly report (power vs significant wave height)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    Hs = np.linspace(0, 8, 80)
    r = m.predict({"Hs_m": Hs})
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Hs, y=r["power_kw"], name="Power (kW)"))
        fig.update_layout(title="EC137 F0a — Attenuator wave-power curve",
                          xaxis_title="Significant wave height Hs (m)", yaxis_title="Power (kW)")
        fig.write_html("simulation_report.html")
        print("wrote simulation_report.html")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for h, p in zip(Hs[::10], r["power_kw"][::10]):
            print(f"  Hs={h:4.2f} m  P={p:7.2f} kW")


if __name__ == "__main__":
    main()
