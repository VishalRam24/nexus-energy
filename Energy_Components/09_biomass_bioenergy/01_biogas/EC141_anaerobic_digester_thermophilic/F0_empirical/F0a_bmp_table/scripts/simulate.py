"""Optional Plotly report for EC141 F0a BMP lookup."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(45, 66))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            print(f"{fs:16s} {m.predict({'feedstock': fs})['bmp_L_CH4_kgVS']:.1f} L/kgVS")
        return
    fig = go.Figure()
    for fs in m.table.feedstocks:
        ys = [m.predict({"feedstock": fs, "temperature_degC": T})["bmp_L_CH4_kgVS"] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=fs, mode="lines"))
    fig.update_layout(title="EC141 F0a BMP vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="L CH4 / kg VS")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
