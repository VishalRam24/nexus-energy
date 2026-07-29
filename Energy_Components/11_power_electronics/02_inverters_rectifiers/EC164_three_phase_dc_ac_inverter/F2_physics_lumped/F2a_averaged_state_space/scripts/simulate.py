"""EC164 -- Three-Phase Inverter -- F2a dq-Frame -- Simulation & HTML Report"""
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
            "P Step: 0 -> 50kW (Q=0)",
            "Q Step: 0 -> 20kvar (P=50kW)",
            "P Ramp: 0 -> 100kW over 0.5s",
            "Power Reversal: +50kW -> -50kW",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: P step
    def p_step(t):
        return 0.0 if t < 0.02 else 50.0
    r1 = model.predict({
        "P_ref_kw": p_step, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.1,
    })
    fig.add_trace(go.Scatter(x=r1["t"]*1e3, y=r1["P"]/1e3, name="P (kW)", line=dict(color="#636EFA", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=r1["t"]*1e3, y=r1["i_d"], name="i_d (A)", line=dict(color="#EF553B", width=2)), row=1, col=1)

    # Panel 2: Q step
    def q_step(t):
        return 0.0 if t < 0.02 else 20.0
    r2 = model.predict({
        "P_ref_kw": 50.0, "Q_ref_kvar": q_step,
        "dt": 1e-5, "duration_s": 0.1,
    })
    fig.add_trace(go.Scatter(x=r2["t"]*1e3, y=r2["Q"]/1e3, name="Q (kvar)", line=dict(color="#00CC96", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=r2["t"]*1e3, y=r2["i_q"], name="i_q (A)", line=dict(color="#FFA15A", width=2)), row=1, col=2)

    # Panel 3: P ramp
    def p_ramp(t):
        return min(t / 0.5, 1.0) * 100.0
    r3 = model.predict({
        "P_ref_kw": p_ramp, "Q_ref_kvar": 0.0,
        "dt": 5e-5, "duration_s": 0.6,
    })
    fig.add_trace(go.Scatter(x=r3["t"]*1e3, y=r3["P"]/1e3, name="P ramp (kW)", line=dict(color="#AB63FA", width=2)), row=2, col=1)

    # Panel 4: Power reversal
    def p_reverse(t):
        return 50.0 if t < 0.05 else -50.0
    r4 = model.predict({
        "P_ref_kw": p_reverse, "Q_ref_kvar": 0.0,
        "dt": 1e-5, "duration_s": 0.15,
    })
    fig.add_trace(go.Scatter(x=r4["t"]*1e3, y=r4["P"]/1e3, name="P reversal (kW)", line=dict(color="#FF6692", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=r4["t"]*1e3, y=r4["i_d"], name="i_d reversal (A)", line=dict(color="#19D3F3", width=2)), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Time (ms)", row=r, col=c)
    fig.update_yaxes(title_text="P (kW) / i_d (A)", row=1, col=1)
    fig.update_yaxes(title_text="Q (kvar) / i_q (A)", row=1, col=2)
    fig.update_yaxes(title_text="P (kW)", row=2, col=1)
    fig.update_yaxes(title_text="P (kW) / i_d (A)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} dq-Frame Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    ss = model.predict_steady_state({"P_ref_kw": 50.0, "Q_ref_kvar": 0.0})
    print(f"\n--- Inverter F2a SS (P=50kW, Q=0) ---")
    print(f"i_d = {ss['i_d_ss']:.3f} A, i_q = {ss['i_q_ss']:.3f} A")
    print(f"P = {ss['P_ss_w']:.1f} W, Q = {ss['Q_ss_var']:.1f} var")


if __name__ == "__main__":
    generate_report()
