"""Optional Plotly report for EC017 F0a — Hydrogen Purifier (PSA). NumPy only; Plotly optional."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    info = m.get_info()
    xk, yk = info["input"], info["output"]
    xq = np.linspace(info["x_range"][0], info["x_range"][1], 100)
    yq = np.asarray(m.predict({xk: xq.tolist()})[yk])
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xq, y=yq, mode="lines", name=yk))
        fig.update_layout(title="%s F0a — %s" % (info["component_id"], info["metric"]),
                          xaxis_title=xk, yaxis_title=yk)
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("Wrote", out)
    except Exception as e:  # noqa: BLE001
        print("Plotly unavailable (%s); printing curve sample instead." % e)
        for xx, yy in list(zip(xq, yq))[::20]:
            print("  %s=%.4g -> %s=%.4g" % (xk, xx, yk, yy))


if __name__ == "__main__":
    main()
