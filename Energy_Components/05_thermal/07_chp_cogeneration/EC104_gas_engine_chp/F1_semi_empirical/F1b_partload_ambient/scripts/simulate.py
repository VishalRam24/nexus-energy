"""EC104 -- Gas Engine CHP -- F1b Part-Load + Ambient -- Simulation & HTML Report"""
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
            "Efficiency vs PLR (Electrical, Thermal, Total)",
            "Efficiency vs Ambient Temperature (full load)",
            "Power & Heat Output vs PLR",
            "Heat-to-Power Ratio vs PLR",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    PLR = np.linspace(0.5, 1.0, 200)

    # Plot 1: Efficiencies vs PLR at 25C
    r = model.predict({"PLR": PLR})
    fig.add_trace(
        go.Scatter(x=PLR, y=r["efficiency_electrical"],
                   name="Electrical", line=dict(width=2, color="blue")),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=PLR, y=r["efficiency_thermal"],
                   name="Thermal", line=dict(width=2, color="red")),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=PLR, y=r["efficiency_total"],
                   name="Total", line=dict(width=3, color="green")),
        row=1, col=1)

    # Plot 2: Efficiency vs ambient temperature at full load
    T_range = np.linspace(-20, 50, 200)
    r_T = model.predict({"PLR": 1.0, "T_ambient": T_range})
    eta_el_arr = np.broadcast_to(r_T["efficiency_electrical"], T_range.shape)
    eta_th_arr = np.broadcast_to(r_T["efficiency_thermal"], T_range.shape)
    eta_tot_arr = np.broadcast_to(r_T["efficiency_total"], T_range.shape)
    fig.add_trace(
        go.Scatter(x=T_range, y=eta_el_arr,
                   name="eta_el vs T", line=dict(width=2, color="blue")),
        row=1, col=2)
    fig.add_trace(
        go.Scatter(x=T_range, y=eta_th_arr,
                   name="eta_th vs T", line=dict(width=2, color="red")),
        row=1, col=2)
    fig.add_trace(
        go.Scatter(x=T_range, y=eta_tot_arr,
                   name="eta_total vs T", line=dict(width=3, color="green")),
        row=1, col=2)
    fig.add_vline(x=25, row=1, col=2, line_dash="dot", line_color="gray",
                  annotation_text="25C ref")

    # Plot 3: Power and heat output vs PLR
    for T_c in [15, 25, 40]:
        r_p = model.predict({"PLR": PLR, "T_ambient": float(T_c)})
        fig.add_trace(
            go.Scatter(x=PLR, y=r_p["power_electrical_kw"],
                       name=f"P_el {T_c}C", line=dict(width=2)),
            row=2, col=1)
        fig.add_trace(
            go.Scatter(x=PLR, y=r_p["heat_recovery_kw"],
                       name=f"Q_th {T_c}C", line=dict(width=2, dash="dash")),
            row=2, col=1)

    # Plot 4: Heat-to-power ratio vs PLR
    for T_c in [15, 25, 40]:
        r_h = model.predict({"PLR": PLR, "T_ambient": float(T_c)})
        fig.add_trace(
            go.Scatter(x=PLR, y=r_h["heat_to_power_ratio"],
                       name=f"HPR {T_c}C", line=dict(width=2)),
            row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Power / Heat (kW)", row=2, col=1)
    fig.update_yaxes(title_text="HPR (-)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Part-Load + Ambient",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
