"""Optional Plotly report for EC058 Flat Plate Solar Collector F0a. Safe if plotly absent."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        print(f"[simulate] plotly/numpy unavailable ({exc}); skipping report.")
        return
    gs = np.linspace(0, 1100, 60)
    key = getattr(m, "irr_key", "irradiance")
    ys = []
    for g in gs:
        try:
            r = m.predict({key: float(g), "cell_temperature": 25.0, "delta_T": 25.0})
        except Exception:
            r = m.predict({key: float(g)})
        ys.append(r.get("power", r.get("power_density")))
    fig = go.Figure(go.Scatter(x=gs, y=ys, mode="lines"))
    fig.update_layout(title="EC058 Flat Plate Solar Collector F0a", xaxis_title="Irradiance [W/m2]",
                      yaxis_title="Output")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    main()
