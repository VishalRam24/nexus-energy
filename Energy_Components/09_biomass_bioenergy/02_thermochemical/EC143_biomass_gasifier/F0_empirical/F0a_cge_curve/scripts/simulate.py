"""Optional Plotly report for EC143 F0a CGE curve."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    ers = [0.15 + 0.01 * i for i in range(31)]
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.curve.feedstocks:
            print(f"{fs:14s} CGE={m.predict({'feedstock': fs})['cold_gas_efficiency']:.3f}")
        return
    fig = go.Figure()
    for fs in m.curve.feedstocks:
        ys = [m.predict({"feedstock": fs, "equivalence_ratio": e})["cold_gas_efficiency"] for e in ers]
        fig.add_trace(go.Scatter(x=ers, y=ys, name=fs, mode="lines"))
    fig.update_layout(title="EC143 F0a CGE vs equivalence ratio",
                      xaxis_title="Equivalence ratio", yaxis_title="Cold gas efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
