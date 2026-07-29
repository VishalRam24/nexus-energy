"""EC168 -- MPPT Controller -- F1b -- Simulation & HTML Report"""
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
            "Efficiency vs Irradiance (steady state, P_mpp proportional to G)",
            "Loss Breakdown vs Irradiance (steady state)",
            "Power Output vs Power Input",
            "Dynamic Efficiency vs Irradiance Ramp Rate (G=800 W/m2)",
        ],
        vertical_spacing=0.14,
    )

    # -- Panel 1: Efficiency vs irradiance (steady state) --
    G_range = np.linspace(50, 1200, 200)
    P_mpp_range = 10000.0 * G_range / 1000.0  # Linear scaling with irradiance
    r = model.predict({"irradiance": G_range, "p_mpp_available": P_mpp_range, "dG_dt": 0.0})
    fig.add_trace(go.Scatter(
        x=G_range, y=r["eta_total"] * 100,
        name="eta_total", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=G_range, y=r["eta_static"] * 100,
        name="eta_static", line=dict(color="#00CC96", width=2, dash="dash"),
    ), row=1, col=1)
    fig.add_hline(y=99, line_dash="dot", line_color="gray",
                  annotation_text="99%", row=1, col=1)

    # -- Panel 2: Loss breakdown --
    fig.add_trace(go.Scatter(
        x=G_range, y=r["p_oscillation_loss_w"],
        name="Oscillation loss", line=dict(color="#EF553B"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=G_range, y=r["p_converter_loss_w"],
        name="Converter loss", line=dict(color="#FFA15A"),
        stackgroup="losses",
    ), row=1, col=2)

    # -- Panel 3: P_out vs P_in --
    fig.add_trace(go.Scatter(
        x=P_mpp_range / 1e3, y=r["p_out_w"] / 1e3,
        name="P_out vs P_mpp", line=dict(color="#19D3F3", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, float(np.max(P_mpp_range)) / 1e3],
        y=[0, float(np.max(P_mpp_range)) / 1e3],
        name="Ideal", line=dict(color="gray", dash="dot"),
        showlegend=False,
    ), row=2, col=1)

    # -- Panel 4: Dynamic efficiency vs ramp rate --
    dG_range = np.linspace(0, 400, 200)
    etas_dyn = []
    for dG in dG_range:
        rd = model.predict({"irradiance": 800.0, "p_mpp_available": 8000.0, "dG_dt": dG})
        etas_dyn.append(float(rd["eta_total"]))
    fig.add_trace(go.Scatter(
        x=dG_range, y=np.array(etas_dyn) * 100,
        name="eta vs dG/dt", line=dict(color="#FF6692", width=2),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="Available MPP Power (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Irradiance Ramp Rate |dG/dt| (W/m2/s)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="Output Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Total Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} P&O Loss Model",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary
    print(f"\n--- MPPT F1b Summary (steady state, dG/dt=0) ---")
    print(f"{'G(W/m2)':>8} {'P_mpp(W)':>9} {'eta_s%':>7} {'eta_d%':>7} {'eta_t%':>7} "
          f"{'P_osc(W)':>9} {'P_conv(W)':>10} {'P_out(W)':>9}")
    for G in [100, 200, 400, 600, 800, 1000, 1200]:
        P_mpp = 10000.0 * G / 1000.0
        rv = model.predict({"irradiance": G, "p_mpp_available": P_mpp, "dG_dt": 0.0})
        print(f"{G:>8} {P_mpp:>9.0f} "
              f"{float(rv['eta_static'])*100:>7.2f} "
              f"{float(rv['eta_dynamic'])*100:>7.2f} "
              f"{float(rv['eta_total'])*100:>7.2f} "
              f"{float(rv['p_oscillation_loss_w']):>9.2f} "
              f"{float(rv['p_converter_loss_w']):>10.2f} "
              f"{float(rv['p_out_w']):>9.1f}")


if __name__ == "__main__":
    generate_report()
