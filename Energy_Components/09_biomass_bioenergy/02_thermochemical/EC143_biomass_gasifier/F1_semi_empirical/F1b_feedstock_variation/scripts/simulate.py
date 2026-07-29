"""EC143 -- Biomass Gasifier -- F1b Feedstock -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import BiomassGasifierF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    gasifier = BiomassGasifierF1b(params)

    feedstocks = list(gasifier.feedstock_db.keys())

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Syngas Composition by Feedstock (ER=0.25)",
            "CO vs Equivalence Ratio",
            "H2 vs Equivalence Ratio",
            "Cold Gas Efficiency by Feedstock",
            "LHV vs ER by Feedstock",
            "Moisture Effect on CGE (Wood)",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    colors = ["firebrick", "steelblue", "green", "purple", "darkorange"]

    # --- 1) Bar: syngas composition at design ER ---
    for fs in feedstocks:
        r = gasifier.predict(fs, 0.25, 0.1)
        comp = r["syngas_composition"]
        fig.add_trace(
            go.Bar(name=fs, x=["CO", "H2", "CO2", "CH4", "N2"],
                   y=[comp["CO"], comp["H2"], comp["CO2"], comp["CH4"], comp["N2"]]),
            row=1, col=1,
        )

    # --- 2 & 3) CO and H2 vs ER ---
    ER_range = np.linspace(0.15, 0.45, 30)
    for i, fs in enumerate(feedstocks):
        CO_vals = []
        H2_vals = []
        for er in ER_range:
            comp = gasifier.syngas_composition(fs, float(er), 0.1)
            CO_vals.append(comp["CO"])
            H2_vals.append(comp["H2"])
        fig.add_trace(
            go.Scatter(x=ER_range, y=CO_vals, name=f"CO-{fs}",
                       line=dict(color=colors[i], width=2)),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=ER_range, y=H2_vals, name=f"H2-{fs}",
                       line=dict(color=colors[i], width=2, dash="dash")),
            row=1, col=3,
        )

    # --- 4) CGE by feedstock ---
    cge_vals = []
    for fs in feedstocks:
        r = gasifier.predict(fs, 0.25, 0.1)
        cge_vals.append(r["cold_gas_efficiency"])
    fig.add_trace(
        go.Bar(x=feedstocks, y=cge_vals, name="CGE", marker_color=colors),
        row=2, col=1,
    )

    # --- 5) LHV vs ER ---
    for i, fs in enumerate(feedstocks):
        lhv_vals = []
        for er in ER_range:
            r = gasifier.predict(fs, float(er), 0.1)
            lhv_vals.append(r["lhv_syngas_mj_nm3"])
        fig.add_trace(
            go.Scatter(x=ER_range, y=lhv_vals, name=f"LHV-{fs}",
                       line=dict(color=colors[i], width=2)),
            row=2, col=2,
        )

    # --- 6) Moisture effect on wood ---
    MC_range = np.linspace(0, 0.5, 30)
    cge_mc = []
    lhv_mc = []
    for mc in MC_range:
        r = gasifier.predict("wood", 0.25, float(mc))
        cge_mc.append(r["cold_gas_efficiency"])
        lhv_mc.append(r["lhv_syngas_mj_nm3"])
    fig.add_trace(
        go.Scatter(x=MC_range * 100, y=cge_mc, name="CGE vs moisture",
                   line=dict(color="firebrick", width=2)),
        row=2, col=3,
    )

    # Axes
    fig.update_xaxes(title_text="Species", row=1, col=1)
    fig.update_xaxes(title_text="Equivalence Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Equivalence Ratio", row=1, col=3)
    fig.update_xaxes(title_text="Feedstock", row=2, col=1)
    fig.update_xaxes(title_text="Equivalence Ratio", row=2, col=2)
    fig.update_xaxes(title_text="Moisture Content (%)", row=2, col=3)
    fig.update_yaxes(title_text="Mole Fraction", row=1, col=1)
    fig.update_yaxes(title_text="CO Fraction", row=1, col=2)
    fig.update_yaxes(title_text="H2 Fraction", row=1, col=3)
    fig.update_yaxes(title_text="CGE [-]", row=2, col=1)
    fig.update_yaxes(title_text="LHV (MJ/Nm3)", row=2, col=2)
    fig.update_yaxes(title_text="CGE [-]", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Feedstock Variation<br>"
            f"<sup>{len(feedstocks)} feedstocks | Ultimate analysis | ER 0.15-0.45 | Moisture correction</sup>"
        ),
        height=900, template="plotly_white", barmode="group",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
