"""EC127 — Gravity Energy Storage — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    m = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Potential Energy vs SOC (E = m·g·h)",
            "Power vs Velocity (with rated-power clamp)",
            "SOC Trajectory: Charge → Hold → Discharge",
            "Round-Trip Energy Balance",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Plot 1: PE vs SOC
    soc = np.linspace(0, 1, 100)
    r = model.predict({"mode": "idle", "soc": soc})
    fig.add_trace(go.Scatter(
        x=soc, y=r["potential_energy_kwh"],
        name="E_pot (kWh)", line=dict(color="royalblue", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=soc, y=r["height_m"],
        name="h (m)", line=dict(color="green", width=2, dash="dash"),
    ), row=1, col=1)

    # Plot 2: Power vs velocity
    v = np.linspace(0, 0.1, 200)
    r_c = model.predict({"mode": "charge", "velocity_mps": v})
    r_d = model.predict({"mode": "discharge", "velocity_mps": v})
    fig.add_trace(go.Scatter(
        x=v, y=r_c["power_kw"],
        name="Charge (kW in)", line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=v, y=-r_d["power_kw"],
        name="Discharge (kW out)", line=dict(color="#00CC96", width=2),
    ), row=1, col=2)
    fig.add_hline(y=m.P_rated, row=1, col=2, line_dash="dot", line_color="gray",
                  annotation_text=f"P_rated={m.P_rated:.0f} kW")

    # Plot 3: SOC trajectory
    dt = 0.1
    s = 0.10
    socs = [s]
    times = [0.0]
    P_c = 4_000.0
    P_d = 5_000.0
    t = 0.0
    for _ in range(int(3.0 / dt)):
        s = m.soc_update(s, P_c, dt, "charge")
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(2.0 / dt)):
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(2.5 / dt)):
        s = m.soc_update(s, P_d, dt, "discharge")
        t += dt; socs.append(s); times.append(t)
    fig.add_trace(go.Scatter(
        x=times, y=socs, name="SOC",
        line=dict(color="#FFA15A", width=2),
    ), row=2, col=1)
    fig.add_vrect(x0=0, x1=3, fillcolor="rgba(99,110,250,0.10)", line_width=0,
                  annotation_text="charge", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=3, x1=5, fillcolor="rgba(200,200,200,0.10)", line_width=0,
                  annotation_text="hold", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=5, x1=7.5, fillcolor="rgba(0,204,150,0.10)", line_width=0,
                  annotation_text="discharge", annotation_position="top left", row=2, col=1)

    # Plot 4: Energy balance (kWh, full cycle of 1 kWh in)
    E_in = 1.0
    E_to_pot = E_in * m.charge_efficiency()
    E_out = E_to_pot * m.discharge_efficiency()
    rte = E_out / E_in
    fig.add_trace(go.Bar(
        x=["E_elec_in", "E_pot_stored", "E_elec_out"],
        y=[E_in, E_to_pot, E_out],
        marker_color=["#EF553B", "#636EFA", "#00CC96"],
        text=[f"{E_in:.3f}", f"{E_to_pot:.3f}", f"{E_out:.3f}"],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)
    fig.add_annotation(
        text=f"RTE = {rte*100:.1f}%",
        xref="paper", yref="paper", x=0.85, y=0.42,
        showarrow=False, font=dict(size=12, color="#2ca02c"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#2ca02c", borderwidth=1,
    )

    fig.update_xaxes(title_text="SOC (-)", row=1, col=1)
    fig.update_xaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_yaxes(title_text="Energy (kWh) / Height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="SOC (-)", row=2, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Energy (kWh per kWh in)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Potential Energy Model",
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")
    print(f"\nSummary: RTE = {rte*100:.2f}%   Capacity = {m.energy_capacity_kwh()/1000:.2f} MWh")


if __name__ == "__main__":
    generate_report()
