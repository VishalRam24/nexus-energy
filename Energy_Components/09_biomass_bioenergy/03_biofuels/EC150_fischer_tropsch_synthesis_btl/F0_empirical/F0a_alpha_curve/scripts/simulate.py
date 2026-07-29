"""Optional Plotly report for EC150 F0a FT alpha curve."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(180, 351, 5))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for T in (180, 230, 280, 350):
            print(f"{T} C: {m.predict({'temperature_degC': T})}")
        return
    fig = go.Figure()
    for key in ("alpha", "CO_conversion", "diesel_selectivity"):
        ys = [m.predict({"temperature_degC": T})[key] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=key, mode="lines"))
    fig.update_layout(title="EC150 F0a FT alpha / conversion / diesel cut vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="Value")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
