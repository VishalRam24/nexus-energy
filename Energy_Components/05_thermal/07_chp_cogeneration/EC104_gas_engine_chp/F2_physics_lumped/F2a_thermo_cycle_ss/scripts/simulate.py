"""EC104 -- Gas Engine CHP -- F2a Otto Cycle -- Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Otto Efficiency vs Compression Ratio",
            "Power & Heat vs Fuel Input",
            "T-s Diagram (Qualitative)",
            "Efficiency vs Ambient Temperature",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: Efficiency vs r
    r_range = np.linspace(8, 16, 50)
    eta_otto = [float(model._model.otto_efficiency(r)) for r in r_range]
    fig.add_trace(go.Scatter(x=r_range, y=eta_otto, name="eta_Otto"), row=1, col=1)

    # Panel 2: Power & heat vs fuel input
    fuel_range = np.linspace(100, 2000, 50)
    P_el, Q_ex, Q_jk = [], [], []
    for f in fuel_range:
        res = model.predict({"fuel_input_kw": float(f)})
        P_el.append(float(res["power_electrical_kw"]))
        Q_ex.append(float(res["heat_exhaust_kw"]))
        Q_jk.append(float(res["heat_jacket_kw"]))
    fig.add_trace(go.Scatter(x=fuel_range, y=P_el, name="P_elec"), row=1, col=2)
    fig.add_trace(go.Scatter(x=fuel_range, y=Q_ex, name="Q_exhaust"), row=1, col=2)
    fig.add_trace(go.Scatter(x=fuel_range, y=Q_jk, name="Q_jacket"), row=1, col=2)

    # Panel 3: Cycle temperatures for nominal case
    res = model._model.cycle_temperatures(500.0)
    T_pts = [float(res["T1"]), float(res["T2"]), float(res["T3"]), float(res["T4"]), float(res["T1"])]
    s_pts = [1.0, 1.0, 2.5, 2.5, 1.0]  # qualitative entropy
    fig.add_trace(go.Scatter(x=s_pts, y=T_pts, name="Otto cycle", mode="lines+markers"), row=2, col=1)

    # Panel 4: Efficiency vs ambient T
    T_amb_range = np.linspace(263, 323, 30)
    eta_el, eta_th = [], []
    for T in T_amb_range:
        res = model.predict({"fuel_input_kw": 500.0, "T_ambient_K": float(T)})
        eta_el.append(float(res["eta_electrical"]))
        eta_th.append(float(res["eta_thermal"]))
    fig.add_trace(go.Scatter(x=T_amb_range - 273.15, y=eta_el, name="eta_elec"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_amb_range - 273.15, y=eta_th, name="eta_thermal"), row=2, col=2)

    fig.update_xaxes(title_text="Compression Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Fuel Input (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Entropy (qualitative)", row=2, col=1)
    fig.update_xaxes(title_text="Ambient Temp (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Otto Cycle",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
