"""Optional Plotly report for EC066 F0a power-curve lookup."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    xs = [v * 0.25 for v in range(0, 121)]
    ys = [m.predict({"wind_speed": v})["power_kw"] for v in xs]
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="power"))
        fig.update_layout(title="EC066 F0a power curve",
                          xaxis_title="wind speed (m/s)",
                          yaxis_title="power (kW)")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump instead:", e)
        for v, p in zip(xs[::8], ys[::8]):
            print(f"  v={v:5.1f} -> {p:8.1f} kW")


if __name__ == "__main__":
    main()
