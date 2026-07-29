"""EC157 -- Buck Converter -- F1b -- Simulation & HTML Report"""
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
            "Efficiency vs Load Current (V_in=48V, V_out=12V)",
            "Loss Breakdown vs Load Current",
            "Power Output vs Power Input",
            "Efficiency vs Switching Frequency",
        ],
        vertical_spacing=0.14,
    )

    # -- Panel 1: Efficiency vs load current --
    i_range = np.linspace(0.1, 20.0, 200)
    r = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i_range})
    fig.add_trace(go.Scatter(
        x=i_range, y=r["efficiency"] * 100,
        name="eta (%)", line=dict(color="#636EFA", width=2.5),
    ), row=1, col=1)
    fig.add_hline(y=95, line_dash="dash", line_color="gray",
                  annotation_text="95%", row=1, col=1)

    # -- Panel 2: Loss breakdown (stacked area) --
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_mosfet_cond_w"],
        name="MOSFET cond", line=dict(color="#EF553B"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_diode_cond_w"],
        name="Diode cond", line=dict(color="#FFA15A"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_inductor_w"],
        name="Inductor DCR", line=dict(color="#00CC96"),
        stackgroup="losses",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=i_range, y=r["p_switching_w"],
        name="Switching", line=dict(color="#AB63FA"),
        stackgroup="losses",
    ), row=1, col=2)

    # -- Panel 3: P_out vs P_in --
    v_out = r["v_out"]
    p_out = v_out * i_range
    p_in = p_out + r["p_loss_w"]
    fig.add_trace(go.Scatter(
        x=p_in, y=p_out,
        name="P_out vs P_in", line=dict(color="#19D3F3", width=2),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[0, float(np.max(p_in))], y=[0, float(np.max(p_in))],
        name="Ideal (100%)", line=dict(color="gray", dash="dot"),
        showlegend=False,
    ), row=2, col=1)

    # -- Panel 4: Efficiency vs switching frequency --
    f_sw_range = np.linspace(10e3, 500e3, 100)
    etas_fsw = []
    for f in f_sw_range:
        params = json.loads(json.dumps(model.params))
        params["unit"]["f_sw"]["value"] = f
        from model import BuckConverterF1b
        m = BuckConverterF1b(params)
        eta_val = m.efficiency(48.0, 12.0, 10.0)
        etas_fsw.append(float(eta_val))
    fig.add_trace(go.Scatter(
        x=f_sw_range / 1e3, y=np.array(etas_fsw) * 100,
        name="eta vs f_sw (I=10A)", line=dict(color="#FF6692", width=2),
    ), row=2, col=2)

    fig.update_xaxes(title_text="Load Current (A)", row=1, col=1)
    fig.update_xaxes(title_text="Load Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Input Power (W)", row=2, col=1)
    fig.update_xaxes(title_text="Switching Frequency (kHz)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power Loss (W)", row=1, col=2)
    fig.update_yaxes(title_text="Output Power (W)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Detailed Loss Model",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary table
    print(f"\n--- Buck Converter F1b Summary (V_in=48V, V_out=12V) ---")
    print(f"{'I(A)':>6} {'D':>6} {'eta%':>7} {'P_mos(W)':>9} {'P_dio(W)':>9} "
          f"{'P_sw(W)':>8} {'P_L(W)':>7} {'P_tot(W)':>9}")
    for i in [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
        rv = model.predict({"v_in": 48.0, "v_out_target": 12.0, "i_load": i})
        print(f"{i:>6.1f} {float(rv['duty_cycle']):>6.4f} "
              f"{float(rv['efficiency'])*100:>7.3f} "
              f"{float(rv['p_mosfet_cond_w']):>9.4f} "
              f"{float(rv['p_diode_cond_w']):>9.4f} "
              f"{float(rv['p_switching_w']):>8.4f} "
              f"{float(rv['p_inductor_w']):>7.4f} "
              f"{float(rv['p_loss_w']):>9.4f}")


if __name__ == "__main__":
    generate_report()
