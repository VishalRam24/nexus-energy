"""EC098 -- ORC -- F1b Part-Load + Condenser Ambient -- Simulation & HTML Report"""
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
            "Efficiency vs PLR (various condenser temps)",
            "Efficiency vs Condenser Temperature (full load)",
            "Power Output vs PLR (various heat source temps)",
            "Heat Rate vs PLR (3600/eta)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    PLR = np.linspace(0.3, 1.0, 200)

    # Plot 1: Efficiency vs PLR at different condenser temps
    for T_cond in [20, 30, 40, 50]:
        r = model.predict({"T_heat_source": 150.0, "T_condenser": float(T_cond), "PLR": PLR})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["efficiency"],
                       name=f"T_cond={T_cond}C", line=dict(width=2)),
            row=1, col=1)

    # Plot 2: Efficiency vs condenser temperature at full load
    T_cond_range = np.linspace(15, 55, 200)
    for T_hot in [100, 150, 200, 250]:
        r = model.predict({"T_heat_source": float(T_hot),
                           "T_condenser": T_cond_range, "PLR": 1.0})
        fig.add_trace(
            go.Scatter(x=T_cond_range, y=r["efficiency"],
                       name=f"T_hot={T_hot}C", line=dict(width=2)),
            row=1, col=2)
    fig.add_vline(x=30, row=1, col=2, line_dash="dot", line_color="gray",
                  annotation_text="Design 30C")

    # Plot 3: Power output vs PLR at various heat source temps
    for T_hot in [100, 150, 200]:
        r = model.predict({"T_heat_source": float(T_hot), "T_condenser": 30.0,
                           "PLR": PLR, "heat_input_kw": 500.0})
        fig.add_trace(
            go.Scatter(x=PLR, y=r["power_output_kw"],
                       name=f"P @T_hot={T_hot}C", line=dict(width=2)),
            row=2, col=1)

    # Plot 4: Heat rate (3600/eta) vs PLR
    for T_cond in [20, 30, 40]:
        r = model.predict({"T_heat_source": 150.0, "T_condenser": float(T_cond), "PLR": PLR})
        hr = 3600.0 / np.maximum(r["efficiency"], 1e-6)
        fig.add_trace(
            go.Scatter(x=PLR, y=hr,
                       name=f"HR T_cond={T_cond}C", line=dict(width=2)),
            row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Condenser Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Power Output (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Heat Rate (kJ/kWh)", row=2, col=2)

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
