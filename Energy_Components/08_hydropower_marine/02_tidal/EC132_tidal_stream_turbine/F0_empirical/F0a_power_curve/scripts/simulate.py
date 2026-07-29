"""EC132 F0a — optional Plotly report (power curve vs current speed)."""
import numpy as np
from predict import ComponentModel


def main():
    m = ComponentModel()
    v = np.linspace(0, 4.5, 90)
    r = m.predict({"current_speed_ms": v})
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"], name="Power (kW)"))
        fig.update_layout(title="EC132 F0a — Tidal stream turbine power curve",
                          xaxis_title="Current speed (m/s)", yaxis_title="Power (kW)")
        fig.write_html("simulation_report.html")
        print("wrote simulation_report.html")
    except ImportError:
        print("plotly not installed; numeric summary:")
        for vi, p in zip(v[::10], r["power_kw"][::10]):
            print(f"  v={vi:4.2f} m/s  P={p:7.1f} kW")


if __name__ == "__main__":
    main()
