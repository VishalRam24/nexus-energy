"""EC017 -- PSA -- F1b Temperature-Pressure -- Simulation Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required"); sys.exit(1)

def generate_report():
    model = ComponentModel()
    P_arr = np.linspace(5.0, 80.0, 100)
    T_arr = np.linspace(253.15, 353.15, 100)
    temps = [263.15, 283.15, 298.15, 318.15, 338.15]
    labels = ["-10C","10C","25C","45C","65C"]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Recovery vs Feed Pressure","Specific Energy vs Pressure",
                        "Recovery vs Temperature","Specific Energy vs Temperature"],
        vertical_spacing=0.13, horizontal_spacing=0.10)

    for T, lbl, clr in zip(temps, labels, colors):
        eta = [float(model.predict({"feed_flow_kg_s":0.1,"feed_h2_fraction":0.75,"feed_pressure_bar":float(P),"temperature_K":T})["recovery"]) for P in P_arr]
        W   = [float(model.predict({"feed_flow_kg_s":0.1,"feed_h2_fraction":0.75,"feed_pressure_bar":float(P),"temperature_K":T})["specific_energy_kWh_per_kg"]) for P in P_arr]
        fig.add_trace(go.Scatter(x=P_arr, y=eta, name=lbl, line=dict(color=clr)), row=1, col=1)
        fig.add_trace(go.Scatter(x=P_arr, y=W, name=lbl+" W", line=dict(color=clr, dash="dash"), showlegend=False), row=1, col=2)

    eta_T = [float(model.predict({"feed_flow_kg_s":0.1,"feed_h2_fraction":0.75,"feed_pressure_bar":20.0,"temperature_K":float(T)})["recovery"]) for T in T_arr]
    W_T   = [float(model.predict({"feed_flow_kg_s":0.1,"feed_h2_fraction":0.75,"feed_pressure_bar":20.0,"temperature_K":float(T)})["specific_energy_kWh_per_kg"]) for T in T_arr]
    fig.add_trace(go.Scatter(x=T_arr-273.15, y=eta_T, name="Recovery vs T (20bar)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_arr-273.15, y=W_T, name="Energy vs T (20bar)", line=dict(color="#ff7f0e")), row=2, col=2)

    fig.update_layout(title_text="EC017 PSA F1b Temperature-Pressure Simulation Report", height=700)
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report saved to {out}")

if __name__ == "__main__":
    generate_report()
