"""EC111 -- Diesel Generator -- F2a Diesel Cycle -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Diesel Cycle Efficiency vs Compression Ratio",
            "BSFC and Overall Efficiency vs Load",
            "Load Step Transient (50% -> 100%)",
            "Frequency Response to Load Step",
            "Fuel Rate vs Load",
            "Generator Efficiency vs Load",
        ],
        vertical_spacing=0.10,
    )

    # -- Panel 1: Efficiency vs compression ratio --
    r_c_vals = np.linspace(12, 24, 50)
    for r_co in [1.5, 2.0, 2.5, 3.0]:
        eta_vals = [model._model.diesel_efficiency(rc, r_co) * 100 for rc in r_c_vals]
        fig.add_trace(go.Scatter(
            x=r_c_vals, y=eta_vals, name=f"r_co={r_co}",
            line=dict(width=2),
        ), row=1, col=1)
    fig.update_xaxes(title_text="Compression Ratio r_c", row=1, col=1)
    fig.update_yaxes(title_text="Thermal Efficiency [%]", row=1, col=1)

    # -- Panel 2: BSFC and efficiency vs load --
    loads = np.linspace(0.1, 1.0, 50)
    bsfc_vals = [model._model.bsfc(lf) * 3.6e9 for lf in loads]
    eta_vals = [model.predict_steady_state({"load_fraction": lf})["eta_overall"] * 100 for lf in loads]
    fig.add_trace(go.Scatter(
        x=loads * 100, y=bsfc_vals, name="BSFC",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=loads * 100, y=eta_vals, name="Overall Efficiency",
        line=dict(color="#EF553B", width=2, dash="dash"),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Load [%]", row=1, col=2)
    fig.update_yaxes(title_text="BSFC [g/kWh] / Efficiency [%]", row=1, col=2)

    # -- Panel 3 & 4: Load step transient --
    def load_step(t):
        return 250000.0 if t < 10.0 else 500000.0
    r = model.predict({"P_load": load_step, "dt": 0.05, "duration_s": 30.0})

    fig.add_trace(go.Scatter(
        x=r["t"], y=r["P_elec_W"] / 1000, name="P_elec [kW]",
        line=dict(color="#00CC96", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=r["t"], y=r["P_engine_W"] / 1000, name="P_engine [kW]",
        line=dict(color="#FFA15A", width=2),
    ), row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=1)
    fig.update_yaxes(title_text="Power [kW]", row=2, col=1)

    fig.add_trace(go.Scatter(
        x=r["t"], y=r["frequency_Hz"], name="Frequency [Hz]",
        line=dict(color="#AB63FA", width=2),
    ), row=2, col=2)
    fig.add_hline(y=50.0, line_dash="dot", line_color="red", row=2, col=2)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="Frequency [Hz]", row=2, col=2)

    # -- Panel 5: Fuel rate vs load --
    fuel_rates = [model.predict_steady_state({"load_fraction": lf})["fuel_rate_L_h"] for lf in loads]
    fig.add_trace(go.Scatter(
        x=loads * 100, y=fuel_rates, name="Fuel Rate",
        line=dict(color="#FF6692", width=2),
    ), row=3, col=1)
    fig.update_xaxes(title_text="Load [%]", row=3, col=1)
    fig.update_yaxes(title_text="Fuel Rate [L/h]", row=3, col=1)

    # -- Panel 6: Generator efficiency vs load --
    eta_gen_vals = [model._model.generator_efficiency(lf) * 100 for lf in loads]
    fig.add_trace(go.Scatter(
        x=loads * 100, y=eta_gen_vals, name="eta_gen",
        line=dict(color="#19D3F3", width=2),
    ), row=3, col=2)
    fig.update_xaxes(title_text="Load [%]", row=3, col=2)
    fig.update_yaxes(title_text="Generator Efficiency [%]", row=3, col=2)

    fig.update_layout(
        title_text=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} {info['sub_fidelity']}",
        height=1200, width=1100,
        showlegend=True,
    )

    out_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    generate_report()
