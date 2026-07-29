"""Optional Plotly report for EC192 F0a — Gas Pressure Regulator. Wrapped in try/except."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    info = m.get_info()
    xk, yk = info["input"], info["output"]
    import numpy as np
    xq = np.linspace(info["x_range"][0], info["x_range"][1], 100)
    yq = m.predict({xk: xq.tolist()})[yk]
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=xq, y=yq, mode="lines", name=yk))
        fig.update_layout(title="%s %s — %s" % (info["component_id"],
                          info["component_name"], info["metric"]),
                          xaxis_title=xk, yaxis_title=yk)
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("Wrote", out)
    except Exception as e:  # noqa: BLE001
        print("Plotly unavailable (%s); curve sample:" % e)
        for xi, yi in list(zip(xq, yq))[::20]:
            print("  %s=%.4g -> %s=%.4g" % (xk, xi, yk, yi))


if __name__ == "__main__":
    main()
