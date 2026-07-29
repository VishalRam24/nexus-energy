"""Optional Plotly report for EC149 F0a biodiesel conversion table."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    ffas = [0.0 + 0.25 * i for i in range(41)]
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            print(f"{fs:18s} conv={m.predict({'feedstock': fs})['conversion']:.3f}")
        return
    ys = [m.predict({"feedstock": "soybean_oil", "ffa_pct": f})["conversion"] for f in ffas]
    fig = go.Figure(go.Scatter(x=ffas, y=ys, mode="lines"))
    fig.update_layout(title="EC149 F0a conversion vs FFA",
                      xaxis_title="FFA (%)", yaxis_title="Conversion")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
