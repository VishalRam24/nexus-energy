"""EC078 — Hot Water Tank TES — F1b Stratified — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from model import HotWaterTankF1b
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    base = Path(__file__).parent.parent
    with open(base / "data" / "parameters.json") as f:
        params = json.load(f)
    model = HotWaterTankF1b(params)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Node Temperatures During Charge (1h)",
            "Node Temperatures During Discharge (1h)",
            "Stored Energy Over Full Cycle",
            "Stratification Profile Snapshots",
        ],
        vertical_spacing=0.13,
    )

    # Scenario: charge 2h, then discharge 2h, then standby 2h
    dt = 30.0

    # Phase 1: Charge
    result_charge = model.simulate(
        T_inlet_hot=80.0, T_inlet_cold=15.0,
        flow_charge=0.1, flow_discharge=0.0,
        T_ambient=20.0, duration_s=3600.0, dt=dt,
    )

    T_hist_c = result_charge["T_history"]
    t_axis_c = np.arange(len(T_hist_c)) * dt / 60.0  # minutes
    for node in range(model.N):
        fig.add_trace(
            go.Scatter(x=t_axis_c, y=T_hist_c[:, node],
                       name=f"Node {node+1}", showlegend=(node < 3)),
            row=1, col=1,
        )

    # Phase 2: Discharge from charged state
    result_discharge = model.simulate(
        T_inlet_hot=80.0, T_inlet_cold=15.0,
        flow_charge=0.0, flow_discharge=0.1,
        T_ambient=20.0, duration_s=3600.0,
        T_initial=result_charge["T_nodes"], dt=dt,
    )

    T_hist_d = result_discharge["T_history"]
    t_axis_d = np.arange(len(T_hist_d)) * dt / 60.0
    for node in range(model.N):
        fig.add_trace(
            go.Scatter(x=t_axis_d, y=T_hist_d[:, node],
                       name=f"D-Node {node+1}", showlegend=False),
            row=1, col=2,
        )

    # Stored energy over full charge-discharge cycle
    energy_charge = []
    for t_arr in T_hist_c:
        e = np.sum(model.m_node * model.cp * (t_arr - model.T_min)) / 3.6e6
        energy_charge.append(e)
    energy_discharge = []
    for t_arr in T_hist_d:
        e = np.sum(model.m_node * model.cp * (t_arr - model.T_min)) / 3.6e6
        energy_discharge.append(e)

    t_total = np.concatenate([t_axis_c, t_axis_c[-1] + t_axis_d])
    e_total = np.concatenate([energy_charge, energy_discharge])
    fig.add_trace(
        go.Scatter(x=t_total, y=e_total, name="Stored Energy",
                   line=dict(color="steelblue"), showlegend=False),
        row=2, col=1,
    )

    # Stratification profiles at different times
    snapshots = {
        "Initial": T_hist_c[0],
        "30 min charge": T_hist_c[min(60, len(T_hist_c)-1)],
        "60 min charge": T_hist_c[-1],
        "30 min discharge": T_hist_d[min(60, len(T_hist_d)-1)],
        "60 min discharge": T_hist_d[-1],
    }
    nodes = np.arange(1, model.N + 1)
    for label, T_snap in snapshots.items():
        fig.add_trace(
            go.Scatter(x=nodes, y=T_snap, name=label, mode="lines+markers"),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Time (min)", row=1, col=1)
    fig.update_xaxes(title_text="Time (min)", row=1, col=2)
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_xaxes(title_text="Node (1=top, 10=bottom)", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (degC)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (degC)", row=2, col=2)

    fig.update_layout(
        title="EC078 — Hot Water Tank — F1b Stratified (10-node) | Charge-Discharge Cycle",
        height=850, template="plotly_white",
    )
    out = base / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
