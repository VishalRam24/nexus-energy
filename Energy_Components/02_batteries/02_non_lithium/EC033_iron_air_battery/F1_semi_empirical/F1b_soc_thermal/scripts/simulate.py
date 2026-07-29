"""EC033 -- Iron-Air Battery -- F1b SOC-Thermal -- Simulation Report"""
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
    temps = [253.15, 273.15, 298.15, 313.15, 333.15]
    labels = ["-20C","0C","25C","40C","60C"]
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    soc = np.linspace(0.05, 0.95, 200)
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["V vs SOC (1A discharge)","R vs T","Q vs I (298K)","Capacity vs T"],
        vertical_spacing=0.13, horizontal_spacing=0.10)
    for T, lbl, clr in zip(temps, labels, colors):
        V = model.predict({"soc": soc, "current": 1.0, "temperature": T})["terminal_voltage"]
        fig.add_trace(go.Scatter(x=soc, y=V, name=lbl, line=dict(color=clr)), row=1, col=1)
    T_arr = np.linspace(253.15, 333.15, 100)
    R_arr = [float(model.predict({"soc":0.5,"current":0,"temperature":T})["internal_resistance"]) for T in T_arr]
    fig.add_trace(go.Scatter(x=T_arr-273.15, y=R_arr, name="R(T)"), row=1, col=2)
    I_arr = np.linspace(0.1, 3.0, 100)
    Q_arr = [float(model.predict({"soc":0.5,"current":float(I),"temperature":298.15})["heat_generation"]) for I in I_arr]
    fig.add_trace(go.Scatter(x=I_arr, y=Q_arr, name="Q vs I", line=dict(color="#2ca02c")), row=2, col=1)
    C_arr = [float(model.predict({"soc":0.5,"current":0,"temperature":T})["effective_capacity"]) for T in T_arr]
    fig.add_trace(go.Scatter(x=T_arr-273.15, y=C_arr, name="C(T)", line=dict(color="#d62728")), row=2, col=2)
    fig.update_layout(title_text="EC033 Iron-Air Battery F1b SOC-Thermal Simulation Report", height=700)
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report saved to {out}")

if __name__ == "__main__":
    generate_report()
