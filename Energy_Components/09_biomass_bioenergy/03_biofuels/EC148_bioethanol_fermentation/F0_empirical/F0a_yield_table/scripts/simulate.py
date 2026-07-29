"""Optional Plotly report for EC148 F0a bioethanol yield table."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    Ts = list(range(20, 46))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            print(f"{fs:14s} {m.predict({'feedstock': fs})['ethanol_L_per_tonne']:.1f} L/t")
        return
    fig = go.Figure()
    for fs in m.table.feedstocks:
        ys = [m.predict({"feedstock": fs, "temperature_degC": T})["ethanol_L_per_tonne"] for T in Ts]
        fig.add_trace(go.Scatter(x=Ts, y=ys, name=fs, mode="lines"))
    fig.update_layout(title="EC148 F0a ethanol yield vs temperature",
                      xaxis_title="Temperature (degC)", yaxis_title="L EtOH / tonne feed")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
