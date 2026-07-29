"""EC175 — Induction Motor/Generator — F1a — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    plr = np.linspace(0.05, 1.2, 200)
    r = model.predict({"load_fraction": plr})

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Part-Load Ratio",
            "Power In/Out vs PLR",
            "Losses vs PLR",
            "Slip vs PLR",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Efficiency curve with IE class bands
    fig.add_trace(go.Scatter(
        x=plr, y=r["efficiency"] * 100,
        name="eta(PLR)", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)
    # Mark rated point
    fig.add_trace(go.Scatter(
        x=[1.0], y=[0.917 * 100],
        mode="markers", name="Rated (IE3)",
        marker=dict(color="red", size=10, symbol="star"),
    ), row=1, col=1)
    # IE3 reference band
    fig.add_hrect(y0=90, y1=92.5, fillcolor="#00CC96", opacity=0.15,
                  annotation_text="IE3 band", annotation_position="top right",
                  row=1, col=1)

    # Plot 2: Input/Output power
    fig.add_trace(go.Scatter(
        x=plr, y=r["input_power_kw"],
        name="P_in (electrical)", line=dict(color="#EF553B", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=plr, y=r["output_power_kw"],
        name="P_out (mechanical)", line=dict(color="#00CC96", width=2),
    ), row=1, col=2)

    # Plot 3: Losses
    fig.add_trace(go.Scatter(
        x=plr, y=r["losses_kw"],
        name="Total losses", line=dict(color="#AB63FA", width=2),
        fill="tozeroy", fillcolor="rgba(171,99,250,0.15)",
    ), row=2, col=1)

    # Plot 4: Slip
    fig.add_trace(go.Scatter(
        x=plr, y=r["slip"] * 100,
        name="Slip (%)", line=dict(color="#FFA15A", width=2),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (PLR)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Losses (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Slip (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} IE3 15kW",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary table
    print("\n--- Efficiency Map Summary ---")
    print(f"{'PLR':>6} {'eta(%)':>8} {'P_in(kW)':>10} {'P_out(kW)':>10} {'Loss(kW)':>9} {'Slip(%)':>8}")
    for p in [0.25, 0.50, 0.75, 1.00, 1.10, 1.20]:
        rv = model.predict({"load_fraction": p})
        print(
            f"{p:>6.2f} {float(rv['efficiency'])*100:>8.2f} "
            f"{float(rv['input_power_kw']):>10.3f} "
            f"{float(rv['output_power_kw']):>10.3f} "
            f"{float(rv['losses_kw']):>9.4f} "
            f"{float(rv['slip'])*100:>8.3f}"
        )


if __name__ == "__main__":
    generate_report()
