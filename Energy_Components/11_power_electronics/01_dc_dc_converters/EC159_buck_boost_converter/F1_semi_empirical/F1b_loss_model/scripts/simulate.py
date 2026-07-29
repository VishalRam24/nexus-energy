"""EC159 -- Buck-Boost Converter -- F1b -- Simulation & HTML Report"""
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
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Load Current (Vin=24V, Vout=12V)",
            "Loss Breakdown vs Load Current",
            "Junction Temperature vs Load",
            "Efficiency vs Switching Frequency",
        ],
        vertical_spacing=0.14,
    )

    i_range = np.linspace(0.1, 10.0, 200)
    r = model.predict({"v_in": 24.0, "v_out_target": 12.0, "i_load": i_range})

    # Panel 1: Efficiency
    fig.add_trace(go.Scatter(x=i_range, y=r["efficiency"] * 100,
        name="eta (%)", line=dict(color="#636EFA", width=2.5)), row=1, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="gray",
                  annotation_text="95%", row=1, col=1)

    # Panel 2: Loss breakdown
    fig.add_trace(go.Scatter(x=i_range, y=r["p_mosfet_cond_w"],
        name="MOSFET cond", line=dict(color="#EF553B"), stackgroup="losses"), row=1, col=2)
    fig.add_trace(go.Scatter(x=i_range, y=r["p_diode_cond_w"],
        name="Diode cond", line=dict(color="#FFA15A"), stackgroup="losses"), row=1, col=2)
    fig.add_trace(go.Scatter(x=i_range, y=r["p_inductor_w"],
        name="Inductor DCR", line=dict(color="#00CC96"), stackgroup="losses"), row=1, col=2)
    fig.add_trace(go.Scatter(x=i_range, y=r["p_switching_w"],
        name="Switching", line=dict(color="#AB63FA"), stackgroup="losses"), row=1, col=2)

    # Panel 3: T_j vs load
    fig.add_trace(go.Scatter(x=i_range, y=r["T_j_degC"],
        name="T_j (degC)", line=dict(color="#FF6692", width=2)), row=2, col=1)
    T_a = model.params["unit"]["T_a"]["value"]
    fig.add_hline(y=T_a, line_dash="dot", line_color="blue",
                  annotation_text=f"T_a={T_a}C", row=2, col=1)

    # Panel 4: Efficiency vs f_sw
    f_sw_range = np.linspace(10e3, 500e3, 100)
    etas_fsw = []
    for f in f_sw_range:
        params = json.loads(json.dumps(model.params))
        params["unit"]["f_sw"]["value"] = f
        from model import BuckBoostConverterF1b
        m = BuckBoostConverterF1b(params)
        etas_fsw.append(float(m.efficiency(24.0, 12.0, 5.0)))
    fig.add_trace(go.Scatter(x=f_sw_range / 1e3, y=np.array(etas_fsw) * 100,
        name="eta vs f_sw (I=5A)", line=dict(color="#19D3F3", width=2)), row=2, col=2)

    fig.update_xaxes(title_text="Load Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Load Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Load Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Switching Frequency (kHz)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="T_j (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Detailed Loss Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
