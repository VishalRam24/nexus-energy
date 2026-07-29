"""EC124 — Liquid Air Energy Storage (LAES) — F1a — Simulation & HTML Report"""
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
            "Liquid Mass / Volume vs SOC",
            "Charge / Discharge Power vs Mass Flow",
            "SOC Trajectory: Charge → Hold (boil-off) → Discharge",
            "Round-Trip Energy Balance (per kg liquid air)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Plot 1: liquid mass vs SOC
    soc = np.linspace(0, 1, 100)
    r = model.predict({"mode": "idle", "soc": soc})
    fig.add_trace(go.Scatter(
        x=soc, y=r["liquid_mass_kg"] / 1000,
        name="Liquid mass (t)", line=dict(color="royalblue", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=soc, y=r["tank_volume_m3"],
        name="Tank volume (m³)", line=dict(color="green", width=2, dash="dash"),
        yaxis="y2",
    ), row=1, col=1)

    # Plot 2: power vs mass flow
    m_dot = np.linspace(0, 100, 100)
    r_c = model.predict({"mode": "charge", "m_dot_liquid_kgs": m_dot})
    r_d = model.predict({"mode": "discharge", "m_dot_liquid_kgs": m_dot})
    fig.add_trace(go.Scatter(
        x=m_dot, y=r_c["power_kw"] / 1000,
        name="Charge (MW in)", line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=m_dot, y=-r_d["power_kw"] / 1000,
        name="Discharge (MW out)", line=dict(color="#00CC96", width=2),
    ), row=1, col=2)

    # Plot 3: SOC trajectory: charge 8h, hold 48h (boil-off), discharge 6h
    dt = 0.25
    s = 0.10
    socs = [s]
    times = [0.0]
    P_c = 40_000.0
    P_d = 50_000.0
    t = 0.0
    for _ in range(int(8.0 / dt)):
        s = m.soc_update(s, P_c, dt, "charge")
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(48.0 / dt)):
        s = m.soc_update(s, 0.0, dt, "idle")
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(6.0 / dt)):
        s = m.soc_update(s, P_d, dt, "discharge")
        t += dt; socs.append(s); times.append(t)
    fig.add_trace(go.Scatter(
        x=times, y=socs, name="SOC",
        line=dict(color="#FFA15A", width=2),
    ), row=2, col=1)
    fig.add_vrect(x0=0, x1=8, fillcolor="rgba(99,110,250,0.10)", line_width=0,
                  annotation_text="charge", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=8, x1=56, fillcolor="rgba(200,200,200,0.10)", line_width=0,
                  annotation_text="hold (boil-off)", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=56, x1=62, fillcolor="rgba(0,204,150,0.10)", line_width=0,
                  annotation_text="discharge", annotation_position="top left", row=2, col=1)

    # Plot 4: Energy balance (kWh/kg)
    E_in = m.w_liq / m.eta_liq
    E_out = m.w_disch * m.eta_pump * m.eta_exp * m.eta_gen
    rte = E_out / E_in
    fig.add_trace(go.Bar(
        x=["E_elec_in", "E_elec_out"],
        y=[E_in, E_out],
        marker_color=["#EF553B", "#00CC96"],
        text=[f"{E_in:.3f}", f"{E_out:.3f}"],
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
    fig.update_xaxes(title_text="Liquid mass flow (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_yaxes(title_text="Liquid mass (t) / Volume (m³)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="SOC (-)", row=2, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Specific energy (kWh/kg)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Round-Trip Model",
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")
    print(f"\nSummary: RTE = {rte*100:.2f}%   Capacity = {m.energy_capacity_kwh()/1000:.1f} MWh")


if __name__ == "__main__":
    generate_report()
