"""EC131 F0a — optional Plotly report (mean power vs tidal range)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    h = np.linspace(0, 12, 60)
    r = m.predict({"tidal_range_amplitude_m": h})
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=r["mean_power_mw"], name="Mean power (MW)"))
        fig.update_layout(title="EC131 F0a — Tidal barrage power vs range",
                          xaxis_title="Tidal range amplitude (m)", yaxis_title="Mean power (MW)")
        fig.write_html("simulation_report.html")
        print("wrote simulation_report.html")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for hi, p in zip(h[::10], r["mean_power_mw"][::10]):
            print(f"  h={hi:5.1f} m  P={p:7.2f} MW")


if __name__ == "__main__":
    main()
