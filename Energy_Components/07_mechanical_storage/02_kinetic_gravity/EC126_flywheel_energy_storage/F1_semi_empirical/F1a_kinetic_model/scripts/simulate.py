"""EC126 — Flywheel Energy Storage — F1a Kinetic Model — Simulation & HTML Report"""
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
            "Energy Stored & SOC vs Speed",
            "Self-Discharge over Time (Starting SOC=1.0)",
            "Electrical Power vs Torque at Various Speeds",
            "Round-Trip Efficiency vs Standby Time",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    rpm_range = np.linspace(8000, 16000, 200)

    # Plot 1: Energy and SOC vs speed (dual-axis via two traces + annotation)
    r = model.predict({"speed_rpm": rpm_range})
    fig.add_trace(
        go.Scatter(x=rpm_range, y=r["energy_stored_kwh"], name="Energy (kWh)",
                   line=dict(color="royalblue", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=rpm_range, y=r["soc"] * 25, name="SOC × E_rated (kWh)",
                   line=dict(color="green", width=2, dash="dash"),
                   ),
        row=1, col=1,
    )
    fig.add_vline(x=12000, row=1, col=1, line_dash="dot", line_color="gray",
                  annotation_text="12000 rpm")

    # Plot 2: Self-discharge decay curve starting from SOC=1
    t_hours = np.linspace(0, 12, 200)
    m = model._model
    E0 = m.E_rated  # kWh
    E_decay = E0 * np.exp(-m.k_sd * t_hours)
    SOC_decay = E_decay / m.E_rated

    fig.add_trace(
        go.Scatter(x=t_hours, y=E_decay, name="Energy (kWh)",
                   line=dict(color="royalblue", width=2)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=t_hours, y=SOC_decay * 100, name="SOC (%)",
                   line=dict(color="green", width=2, dash="dash")),
        row=1, col=2,
    )

    # Plot 3: Power vs torque at different speeds
    torque_range = np.linspace(-100, 100, 200)
    for rpm in [8000, 10000, 12000, 14000, 16000]:
        r_p = model.predict({"speed_rpm": float(rpm), "torque_nm": torque_range})
        fig.add_trace(
            go.Scatter(x=torque_range, y=r_p["power_kw"],
                       name=f"{rpm} rpm", line=dict(width=2)),
            row=2, col=1,
        )
    fig.add_hline(y=0, row=2, col=1, line_color="black", line_width=1)
    fig.add_vline(x=0, row=2, col=1, line_color="black", line_width=1)

    # Plot 4: RTE vs standby time
    t_rte = np.linspace(0, 24, 200)
    r_rte = model.predict({"speed_rpm": 12000.0, "time_hours": t_rte})
    fig.add_trace(
        go.Scatter(x=t_rte, y=r_rte["round_trip_efficiency"] * 100,
                   name="RTE (%)", line=dict(color="orange", width=2)),
        row=2, col=2,
    )
    # Mark RTE at t=0 (no standby)
    rte_0 = 0.95 * 0.95 * 100
    fig.add_annotation(x=0, y=rte_0,
                       text=f"RTE(0h) = {rte_0:.1f}%",
                       showarrow=True, arrowhead=2,
                       ax=40, ay=-30,
                       row=2, col=2)
    # 1 hour mark
    rte_1h = float(model.predict({"speed_rpm": 12000.0, "time_hours": 1.0})["round_trip_efficiency"]) * 100
    fig.add_annotation(x=1, y=rte_1h,
                       text=f"RTE(1h) = {rte_1h:.1f}%",
                       showarrow=True, arrowhead=2,
                       ax=40, ay=-20,
                       row=2, col=2)

    # Axes
    fig.update_xaxes(title_text="Speed (rpm)", row=1, col=1)
    fig.update_xaxes(title_text="Standby Time (h)", row=1, col=2)
    fig.update_xaxes(title_text="Torque (N·m)", row=2, col=1)
    fig.update_xaxes(title_text="Standby Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="Energy / Proxy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (kWh) / SOC (%)", row=1, col=2)
    fig.update_yaxes(title_text="Electrical Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Round-Trip Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Kinetic Model",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
