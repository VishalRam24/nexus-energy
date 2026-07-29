"""EC153 -- Binary Geothermal -- F1b Part-Load Ambient -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import BinaryGeothermalF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    geo = BinaryGeothermalF1b(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Power vs Part-Load Ratio",
            "Power vs Ambient Temperature",
            "Resource Decline Over 30 Years",
            "Part-Load Efficiency Curve",
            "Combined Derating: PLR + Ambient",
            "Annual Energy vs Years",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # --- 1) Power vs PLR ---
    PLR_range = np.linspace(0.3, 1.0, 30)
    P_plr = []
    for plr in PLR_range:
        r = geo.predict(150.0, 80.0, 30.0, float(plr), 0.0)
        P_plr.append(r["power_output_kw"])
    fig.add_trace(
        go.Scatter(x=PLR_range, y=P_plr, name="Power vs PLR",
                   line=dict(color="firebrick", width=2)),
        row=1, col=1,
    )

    # --- 2) Power vs ambient temperature ---
    T_amb_range = np.linspace(-10, 45, 30)
    P_amb = []
    for T_a in T_amb_range:
        r = geo.predict(150.0, 80.0, float(T_a), 1.0, 0.0)
        P_amb.append(r["power_output_kw"])
    fig.add_trace(
        go.Scatter(x=T_amb_range, y=P_amb, name="Power vs T_amb",
                   line=dict(color="steelblue", width=2)),
        row=1, col=2,
    )

    # --- 3) Resource decline ---
    years_range = np.linspace(0, 30, 50)
    P_years = []
    f_res = []
    for y in years_range:
        r = geo.predict(150.0, 80.0, 30.0, 1.0, float(y))
        P_years.append(r["power_output_kw"])
        f_res.append(r["resource_factor"])
    fig.add_trace(
        go.Scatter(x=years_range, y=P_years, name="Power output",
                   line=dict(color="darkorange", width=2)),
        row=1, col=3,
    )
    fig.add_trace(
        go.Scatter(x=years_range, y=f_res, name="Resource factor",
                   line=dict(color="green", width=2, dash="dash")),
        row=1, col=3,
    )

    # --- 4) Part-load efficiency curve ---
    eta_plr = []
    for plr in PLR_range:
        eta_plr.append(geo.f_plr(float(plr)))
    fig.add_trace(
        go.Scatter(x=PLR_range, y=eta_plr, name="eta_ratio (PLR)",
                   line=dict(color="purple", width=2)),
        row=2, col=1,
    )

    # --- 5) Combined derating surface (PLR x T_amb) ---
    for T_a in [10, 25, 35, 45]:
        P_combined = []
        for plr in PLR_range:
            r = geo.predict(150.0, 80.0, float(T_a), float(plr), 0.0)
            P_combined.append(r["power_output_kw"])
        fig.add_trace(
            go.Scatter(x=PLR_range, y=P_combined, name=f"T_amb={T_a} degC",
                       line=dict(width=2)),
            row=2, col=2,
        )

    # --- 6) Annual energy vs years ---
    annual_energy = []
    for y in years_range:
        # Average over 4 seasons (different ambient temps)
        E_yr = 0
        for T_a in [5, 20, 30, 15]:  # winter, spring, summer, fall
            r = geo.predict(150.0, 80.0, float(T_a), 0.9, float(y))
            E_yr += r["power_output_kw"] * 8760 / 4 / 1000  # MWh
        annual_energy.append(E_yr)
    fig.add_trace(
        go.Scatter(x=years_range, y=annual_energy, name="Annual energy (MWh)",
                   line=dict(color="firebrick", width=2)),
        row=2, col=3,
    )

    # Axes
    fig.update_xaxes(title_text="PLR", row=1, col=1)
    fig.update_xaxes(title_text="T_ambient (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Years", row=1, col=3)
    fig.update_xaxes(title_text="PLR", row=2, col=1)
    fig.update_xaxes(title_text="PLR", row=2, col=2)
    fig.update_xaxes(title_text="Years", row=2, col=3)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=1, col=3)
    fig.update_yaxes(title_text="eta_ratio", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=2)
    fig.update_yaxes(title_text="Energy (MWh/yr)", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Part-Load + Ambient<br>"
            f"<sup>5 MW | eta=0.12 | 1.5%/yr decline | Air-cooled condenser derating</sup>"
        ),
        height=900, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
