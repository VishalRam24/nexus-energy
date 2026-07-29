"""Optional Plotly report for EC147 F0a HTL bio-crude yield table."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(250, 401, 10))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            print(f"{fs:28s} {m.predict({'feedstock': fs})['biocrude_yield']:.3f}")
        return
    fig = go.Figure()
    for fs in m.table.feedstocks:
        ys = [m.predict({"feedstock": fs, "temperature_degC": T})["biocrude_yield"] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=fs, mode="lines"))
    fig.update_layout(title="EC147 F0a bio-crude yield vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="Bio-crude mass yield")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
