"""Optional Plotly report for EC144 F0a part-load efficiency curve."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plrs = [0.2 + 0.02 * i for i in range(41)]
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for p in (0.2, 0.5, 0.8, 1.0):
            print(f"PLR={p}: eta_total={m.predict({'PLR': p})['eta_total']:.3f}")
        return
    fig = go.Figure()
    for key in ("eta_electrical", "eta_thermal", "eta_total"):
        ys = [m.predict({"PLR": p})[key] for p in plrs]
        fig.add_trace(go.Scatter(x=plrs, y=ys, name=key, mode="lines"))
    fig.update_layout(title="EC144 F0a CHP efficiency vs part-load",
                      xaxis_title="Part-load ratio", yaxis_title="Efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
