"""EC160 -- Isolated DC-DC Flyback -- F2a -- Optional Plotly simulation report."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()
    m = cm._model

    # 1) Dynamic start-up transient
    r = cm.predict({"v_in": 48.0, "duty_cycle": 0.5, "R_load": 1.2,
                    "dt": 1e-6, "duration_s": 0.02})

    # 2) DC gain & efficiency sweep over duty
    ds = np.linspace(0.05, 0.85, 60)
    v_ideal = np.array([m.ideal_gain(48.0, d) for d in ds])
    v_real = np.array([m.steady_state(48.0, d, 1.2)["v_out_ss"] for d in ds])
    eta = np.array([m.efficiency(48.0, d, 1.2) for d in ds])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly unavailable ({e}); printing summary only.")
        print(f"  Startup V_out final = {r['v_out'][-1]:.3f} V")
        print(f"  d=0.5: V_ideal={m.ideal_gain(48.0,0.5):.3f} V, "
              f"V_real={m.steady_state(48.0,0.5,1.2)['v_out_ss']:.3f} V, "
              f"eta={m.efficiency(48.0,0.5,1.2):.4f}")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Start-up transient: v_out(t)", "Magnetizing current i_m(t)",
        "DC gain vs duty (ideal vs with losses)", "Efficiency vs duty"))

    fig.add_trace(go.Scatter(x=r["t"]*1e3, y=r["v_out"], name="v_out"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t"]*1e3, y=r["i_m"], name="i_m"), 1, 2)
    fig.add_trace(go.Scatter(x=ds, y=v_ideal, name="ideal"), 2, 1)
    fig.add_trace(go.Scatter(x=ds, y=v_real, name="with losses"), 2, 1)
    fig.add_trace(go.Scatter(x=ds, y=eta, name="efficiency"), 2, 2)

    fig.update_xaxes(title_text="t [ms]", row=1, col=1)
    fig.update_xaxes(title_text="t [ms]", row=1, col=2)
    fig.update_xaxes(title_text="duty d", row=2, col=1)
    fig.update_xaxes(title_text="duty d", row=2, col=2)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=1, col=2)
    fig.update_yaxes(title_text="V_out", row=2, col=1)
    fig.update_yaxes(title_text="eta", row=2, col=2)
    fig.update_layout(title="EC160 Flyback F2a -- Averaged State-Space Model",
                      height=720, width=1000)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
