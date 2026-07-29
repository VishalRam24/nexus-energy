"""Optional Plotly report for EC146 F0a torrefaction mass-yield table."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(200, 301, 5))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for T in (200, 250, 300):
            print(f"{T} C: {m.predict({'temperature_degC': T})}")
        return
    fig = go.Figure()
    for key in ("mass_yield", "EDR", "energy_yield"):
        ys = [m.predict({"temperature_degC": T})[key] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=key, mode="lines"))
    fig.update_layout(title="EC146 F0a torrefaction yields vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="Yield / ratio")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
