"""Optional Plotly report for EC140 F0a yield lookup."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(25, 56))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            y = m.predict({"feedstock": fs})["ch4_yield_m3_kgVS"]
            print(f"{fs:16s} {y:.3f} m3/kgVS")
        return
    fig = go.Figure()
    for fs in m.table.feedstocks:
        ys = [m.predict({"feedstock": fs, "temperature_degC": T})["ch4_yield_m3_kgVS"] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=fs, mode="lines"))
    fig.update_layout(title="EC140 F0a CH4 yield vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="m3 CH4 / kg VS")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
