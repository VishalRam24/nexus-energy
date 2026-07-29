"""EC014 -- Metal Hydride H2 Storage -- F1b van't Hoff Thermal -- Simulation Report"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required: pip install plotly")
    sys.exit(1)


def generate_report():
    model = ComponentModel()
    temps = [253.15, 273.15, 298.15, 323.15, 353.15]
    labels = ["-20C", "0C", "25C", "50C", "80C"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    soc_arr = np.linspace(0.0, 1.0, 100)

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Desorption Plateau Pressure vs SOC",
                        "Absorption vs Desorption (25C)",
                        "Sorption Rate vs Pressure (298 K)",
                        "Thermal Balance: dT/dt vs Time (absorption)"],
        vertical_spacing=0.13, horizontal_spacing=0.10)

    # Panel 1: P_eq vs SOC at different T
    for T, lbl, clr in zip(temps, labels, colors):
        P_arr = [float(model.predict({"temperature": T, "pressure_bar": 1.0, "soc": float(s), "mode": "desorption"})["plateau_pressure_bar"]) for s in soc_arr]
        fig.add_trace(go.Scatter(x=soc_arr, y=P_arr, name=lbl, line=dict(color=clr)), row=1, col=1)

    # Panel 2: Absorption vs desorption at 25C
    P_abs = [float(model.predict({"temperature": 298.15, "pressure_bar": 1.0, "soc": float(s), "mode": "absorption"})["plateau_pressure_bar"]) for s in soc_arr]
    P_des = [float(model.predict({"temperature": 298.15, "pressure_bar": 1.0, "soc": float(s), "mode": "desorption"})["plateau_pressure_bar"]) for s in soc_arr]
    fig.add_trace(go.Scatter(x=soc_arr, y=P_abs, name="Absorption (25C)", line=dict(color="#2ca02c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=soc_arr, y=P_des, name="Desorption (25C)", line=dict(color="#d62728", dash="dash")), row=1, col=2)

    # Panel 3: Sorption rate vs P
    P_range = np.linspace(0.5, 30.0, 100)
    rates = [float(model.predict({"temperature": 298.15, "pressure_bar": float(P), "soc": 0.3, "mode": "absorption"})["sorption_rate_kg_s"]) * 1000 for P in P_range]
    fig.add_trace(go.Scatter(x=P_range, y=rates, name="Rate (298K, SOC=0.3)", line=dict(color="#1f77b4")), row=2, col=1)

    # Panel 4: Thermal dTdt
    soc_sim = np.linspace(0.0, 0.9, 100)
    dTdt = [float(model.predict({"temperature": 298.15, "pressure_bar": 20.0, "soc": float(s), "mode": "absorption", "T_amb_K": 293.15})["dTdt_K_s"]) for s in soc_sim]
    fig.add_trace(go.Scatter(x=soc_sim, y=dTdt, name="dT/dt (absorption)", line=dict(color="#9467bd")), row=2, col=2)

    fig.update_layout(title_text="EC014 Metal Hydride H2 F1b Thermal Simulation Report", height=700)
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out))
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
