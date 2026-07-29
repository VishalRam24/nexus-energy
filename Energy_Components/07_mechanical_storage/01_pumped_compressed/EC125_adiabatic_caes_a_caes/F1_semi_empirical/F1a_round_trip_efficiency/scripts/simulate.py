"""EC125 — Adiabatic CAES (A-CAES) — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# EC123 parameters for RTE comparison bar chart
DIABATIC_RTE = 0.47   # representative diabatic CAES RTE (Huntorf-class with fuel)
DIABATIC_LABEL = "Diabatic CAES (EC123)"


def generate_report():
    model = ComponentModel()
    info  = model.get_info()
    m     = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Cavern Pressure vs SOC",
            "Charge / Discharge Power vs Air Mass Flow",
            "SOC Trajectory: Charge → Hold → Discharge",
            "Round-Trip Efficiency: A-CAES vs Diabatic CAES",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # Plot 1: Cavern pressure vs SOC
    soc = np.linspace(0, 1, 100)
    r   = model.predict({"mode": "idle", "soc": soc})
    fig.add_trace(go.Scatter(
        x=soc, y=r["cavern_pressure_pa"] / 1e5,
        name="Pressure (bar)", line=dict(color="royalblue", width=2),
    ), row=1, col=1)
    fig.add_hline(y=m.p_max / 1e5, row=1, col=1, line_dash="dot", line_color="red",
                  annotation_text=f"p_max={m.p_max/1e5:.0f} bar")
    fig.add_hline(y=m.p_min / 1e5, row=1, col=1, line_dash="dot", line_color="orange",
                  annotation_text=f"p_min={m.p_min/1e5:.0f} bar")

    # Plot 2: power and TES heat vs mass flow
    m_dot = np.linspace(0, 500, 100)
    r_c   = model.predict({"mode": "charge",    "m_dot_air": m_dot})
    r_d   = model.predict({"mode": "discharge", "m_dot_air": m_dot})
    fig.add_trace(go.Scatter(
        x=m_dot, y=r_c["power_kw"] / 1000,
        name="Charge (MW in)", line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=m_dot, y=-r_d["power_kw"] / 1000,
        name="Discharge (MW out)", line=dict(color="#00CC96", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=m_dot, y=r_c["tes_heat_kw"] / 1000,
        name="TES heat stored (MW_th)", line=dict(color="#FFA15A", width=2, dash="dash"),
    ), row=1, col=2)

    # Plot 3: SOC trajectory (charge 6 h → hold 4 h → discharge 4 h)
    dt = 0.25
    s  = 0.10
    socs  = [s]
    times = [0.0]
    P_charge = 60_000.0   # kW
    P_disch  = 100_000.0  # kW
    t = 0.0
    for _ in range(int(6.0 / dt)):
        s = m.soc_update(s, P_charge, dt, "charge")
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(4.0 / dt)):
        t += dt; socs.append(s); times.append(t)
    for _ in range(int(4.0 / dt)):
        s = m.soc_update(s, P_disch, dt, "discharge")
        t += dt; socs.append(s); times.append(t)
    fig.add_trace(go.Scatter(
        x=times, y=socs, name="SOC (A-CAES)",
        line=dict(color="#636EFA", width=2),
    ), row=2, col=1)
    fig.add_vrect(x0=0, x1=6, fillcolor="rgba(99,110,250,0.10)", line_width=0,
                  annotation_text="charge", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=6, x1=10, fillcolor="rgba(200,200,200,0.10)", line_width=0,
                  annotation_text="hold", annotation_position="top left", row=2, col=1)
    fig.add_vrect(x0=10, x1=14, fillcolor="rgba(0,204,150,0.10)", line_width=0,
                  annotation_text="discharge", annotation_position="top left", row=2, col=1)

    # Plot 4: RTE comparison bar chart
    acaes_rte = m.round_trip_efficiency()
    fig.add_trace(go.Bar(
        x=["A-CAES (EC125)", DIABATIC_LABEL],
        y=[acaes_rte * 100, DIABATIC_RTE * 100],
        marker_color=["#00CC96", "#EF553B"],
        text=[f"{acaes_rte*100:.1f}%", f"{DIABATIC_RTE*100:.1f}%"],
        textposition="outside",
        showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(title_text="SOC (-)", row=1, col=1)
    fig.update_xaxes(title_text="Air mass flow (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (h)", row=2, col=1)
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_yaxes(title_text="Cavern pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="SOC (-)", row=2, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Round-Trip Efficiency (%)", row=2, col=2, range=[0, 85])

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} (No Fuel)",
        height=820,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")
    print(f"\nA-CAES RTE       = {acaes_rte*100:.2f}%")
    print(f"Diabatic CAES RTE = {DIABATIC_RTE*100:.1f}%  (reference)")
    print(f"RTE improvement   = {(acaes_rte - DIABATIC_RTE)*100:.1f} percentage points")
    print(f"Energy capacity   = {m.energy_capacity_kwh()/1000:.1f} MWh")
    print(f"Fuel input        = 0 kW  (adiabatic — no supplemental fuel)")


if __name__ == "__main__":
    generate_report()
