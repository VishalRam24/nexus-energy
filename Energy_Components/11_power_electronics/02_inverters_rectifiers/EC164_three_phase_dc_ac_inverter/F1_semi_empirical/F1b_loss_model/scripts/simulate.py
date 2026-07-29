"""EC164 -- Three-Phase Inverter -- F1b -- Simulation & HTML Report"""
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
            "Efficiency vs Load Power (V_dc=800V, m=0.9, PF=1)",
            "Loss Breakdown vs Load Power",
            "Power Output vs Power Input",
            "Efficiency Map: Modulation Index vs Power Factor (P=80kW)",
        ],
        vertical_spacing=0.14,
        specs=[[{}, {}], [{}, {"type": "heatmap"}]],
    )

    # -- Panel 1: Efficiency vs load power --
    p_range = np.linspace(1000, 100000, 200)
    etas = []
    losses_igbt_cond = []
    losses_igbt_sw = []
    losses_diode_cond = []
    losses_diode_rr = []
    total_losses = []
    for p in p_range:
        r = model.predict({"v_dc": 800.0, "p_load": p, "m": 0.9, "power_factor": 1.0})
        etas.append(float(r["efficiency"]))
        losses_igbt_cond.append(float(r["p_igbt_cond_w"]))
        losses_igbt_sw.append(float(r["p_igbt_sw_w"]))
        losses_diode_cond.append(float(r["p_diode_cond_w"]))
        losses_diode_rr.append(float(r["p_diode_rr_w"]))
        total_losses.append(float(r["p_loss_w"]))

    etas = np.array(etas)
    fig.add_trace(go.Scatter(
        x=p_range / 1e3, y=etas * 100,
        name="eta (%)", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)
    fig.add_hline(y=97, line_dash="dash", line_color="gray",
                  annotation_text="97%", row=1, col=1)

    # -- Panel 2: Loss breakdown --
    fig.add_trace(go.Scatter(
        x=p_range / 1e3, y=losses_igbt_cond,
        name="IGBT cond", line=dict(color="#EF553B"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p_range / 1e3, y=losses_igbt_sw,
        name="IGBT switching", line=dict(color="#FFA15A"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p_range / 1e3, y=losses_diode_cond,
        name="Diode cond", line=dict(color="#00CC96"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=p_range / 1e3, y=losses_diode_rr,
        name="Diode recovery", line=dict(color="#AB63FA"),
        stackgroup="losses",
    ), row=1, col=2)

    # -- Panel 3: P_out vs P_in --
    p_in = p_range + np.array(total_losses)
    fig.add_trace(go.Scatter(
        x=p_in / 1e3, y=p_range / 1e3,
        name="P_out vs P_in", line=dict(color="#19D3F3", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, float(np.max(p_in)) / 1e3], y=[0, float(np.max(p_in)) / 1e3],
        name="Ideal", line=dict(color="gray", dash="dot"),
        showlegend=False,
    ), row=2, col=1)

    # -- Panel 4: Efficiency map (m vs PF) --
    m_vals = np.linspace(0.3, 1.0, 30)
    pf_vals = np.linspace(0.5, 1.0, 30)
    eta_map = np.zeros((len(pf_vals), len(m_vals)))
    for i, pf in enumerate(pf_vals):
        for j, m in enumerate(m_vals):
            r = model.predict({"v_dc": 800.0, "p_load": 80000.0, "m": m, "power_factor": pf})
            eta_map[i, j] = float(r["efficiency"]) * 100
    fig.add_trace(go.Heatmap(
        x=m_vals, y=pf_vals, z=eta_map,
        colorscale="Viridis", colorbar=dict(title="eta (%)"),
        name="Efficiency Map",
    ), row=2, col=2)

    fig.update_xaxes(title_text="Load Power (kW)", row=1, col=1)
    fig.update_xaxes(title_text="Load Power (kW)", row=1, col=2)
    fig.update_xaxes(title_text="Input Power (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Modulation Index m", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="Output Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Power Factor", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} IGBT/Diode Loss Model",
        height=900,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary
    print(f"\n--- Three-Phase Inverter F1b Summary (V_dc=800V, m=0.9, PF=1.0) ---")
    print(f"{'P(kW)':>7} {'eta%':>7} {'P_igbt_c':>9} {'P_igbt_sw':>10} "
          f"{'P_diode_c':>10} {'P_diode_rr':>11} {'P_total':>8}")
    for p in [10000, 25000, 50000, 75000, 100000]:
        rv = model.predict({"v_dc": 800.0, "p_load": p, "m": 0.9, "power_factor": 1.0})
        print(f"{p/1e3:>7.0f} {float(rv['efficiency'])*100:>7.3f} "
              f"{float(rv['p_igbt_cond_w']):>9.1f} "
              f"{float(rv['p_igbt_sw_w']):>10.1f} "
              f"{float(rv['p_diode_cond_w']):>10.1f} "
              f"{float(rv['p_diode_rr_w']):>11.1f} "
              f"{float(rv['p_loss_w']):>8.1f}")


if __name__ == "__main__":
    generate_report()
