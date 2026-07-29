"""EC079 -- Molten Salt TES -- F1b Stratified -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import MoltenSaltTESF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    tes = MoltenSaltTESF1b(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Node Temperatures After 1h Charge",
            "Node Temperatures After 1h Discharge",
            "Salt Properties vs Temperature",
            "12h Charge/Discharge Cycle - Energy",
            "12h Cycle - Node Temperature Profiles",
            "Stratification Thermocline Evolution",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # --- 1) Charge: uniform cold start, charge 1h ---
    T_cold_init = [290.0] * 10
    r_charge = tes.predict(565.0, 290.0, 500.0, "charge", 25.0, 3600.0, T_cold_init)
    nodes = list(range(10))
    fig.add_trace(
        go.Scatter(x=nodes, y=r_charge["T_nodes"], name="After 1h charge",
                   mode="lines+markers", line=dict(color="firebrick", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=nodes, y=T_cold_init, name="Initial (cold)",
                   mode="lines+markers", line=dict(color="steelblue", dash="dash")),
        row=1, col=1,
    )

    # --- 2) Discharge: uniform hot start, discharge 1h ---
    T_hot_init = [565.0] * 10
    r_discharge = tes.predict(565.0, 290.0, 500.0, "discharge", 25.0, 3600.0, T_hot_init)
    fig.add_trace(
        go.Scatter(x=nodes, y=r_discharge["T_nodes"], name="After 1h discharge",
                   mode="lines+markers", line=dict(color="steelblue", width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=nodes, y=T_hot_init, name="Initial (hot)",
                   mode="lines+markers", line=dict(color="firebrick", dash="dash")),
        row=1, col=2,
    )

    # --- 3) Salt properties vs T ---
    T_range = np.linspace(220, 600, 200)
    rho_vals = tes.rho(T_range)
    cp_vals = tes.cp(T_range)
    fig.add_trace(
        go.Scatter(x=T_range, y=rho_vals, name="Density [kg/m3]",
                   line=dict(color="green", width=2)),
        row=1, col=3,
    )
    fig.add_trace(
        go.Scatter(x=T_range, y=cp_vals, name="Specific heat [J/(kg*K)]",
                   line=dict(color="purple", width=2, dash="dash")),
        row=1, col=3,
    )

    # --- 4 & 5) 12h cycle: 6h charge, 6h discharge ---
    N_steps = 12
    modes = ["charge"] * 6 + ["discharge"] * 6
    flows = np.array([500.0] * 6 + [500.0] * 6)
    T_ch = np.full(N_steps, 565.0)
    T_dis = np.full(N_steps, 290.0)
    T_init = np.full(10, 290.0)

    sim = tes.simulate(T_init, modes, flows, T_ch, T_dis, dt_step_s=3600.0)

    hours = np.arange(N_steps)
    fig.add_trace(
        go.Scatter(x=hours, y=sim["stored_energy_kwh"], name="Stored Energy [kWh]",
                   line=dict(color="darkorange", width=2)),
        row=2, col=1,
    )

    # Node temperature profiles over time
    colors = [f"hsl({h},70%,50%)" for h in np.linspace(0, 240, 10)]
    for node_idx in [0, 4, 9]:
        T_node = [sim["T_nodes_history"][t][node_idx] for t in range(N_steps)]
        fig.add_trace(
            go.Scatter(x=hours, y=T_node, name=f"Node {node_idx}",
                       line=dict(width=2)),
            row=2, col=2,
        )

    # --- 6) Thermocline at various times ---
    for t_idx in [0, 2, 5, 8, 11]:
        label = f"t={t_idx}h"
        T_profile = sim["T_nodes_history"][t_idx]
        fig.add_trace(
            go.Scatter(x=list(range(10)), y=T_profile, name=label,
                       mode="lines+markers"),
            row=2, col=3,
        )

    # Axes
    fig.update_xaxes(title_text="Node (0=top, 9=bottom)", row=1, col=1)
    fig.update_xaxes(title_text="Node (0=top, 9=bottom)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=3)
    fig.update_xaxes(title_text="Hour", row=2, col=1)
    fig.update_xaxes(title_text="Hour", row=2, col=2)
    fig.update_xaxes(title_text="Node", row=2, col=3)
    fig.update_yaxes(title_text="Temperature (degC)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=3)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (degC)", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Stratified 10-Node<br>"
            f"<sup>Solar salt | rho(T)=2090-0.636T | cp(T)=1443+0.172T | "
            f"1000 m3 | 14 m tall | 290-565 degC</sup>"
        ),
        height=900,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
