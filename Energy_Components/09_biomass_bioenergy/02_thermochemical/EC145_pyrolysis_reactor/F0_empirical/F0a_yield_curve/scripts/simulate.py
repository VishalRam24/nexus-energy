"""Optional Plotly report for EC145 F0a pyrolysis yield curve."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(300, 701, 10))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for T in (300, 500, 700):
            print(f"{T} C: {m.predict({'temperature_degC': T})}")
        return
    fig = go.Figure()
    for key in ("bio_oil_frac", "char_frac", "gas_frac"):
        ys = [m.predict({"temperature_degC": T})[key] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=key, mode="lines"))
    fig.update_layout(title="EC145 F0a pyrolysis yields vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="Mass fraction")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
