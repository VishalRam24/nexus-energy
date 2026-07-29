"""EC073 — Shell-and-Tube HX — F1b Fouling — Simulation Scenarios"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

BASE = Path(__file__).parent.parent


def run():
    model = ComponentModel()

    Rf_arr   = np.linspace(0.0, 0.002, 50)
    mdot_arr = np.linspace(0.5, 20.0, 50)

    # Scenario 1: Fouling sweep
    r_rf = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                          "m_dot_hot": 5.0, "m_dot_cold": 5.0,
                          "Rf_shell": Rf_arr, "Rf_tube": Rf_arr})

    # Scenario 2: Flow rate sweep at fixed fouling
    r_md = model.predict({"T_h_in": 90.0, "T_c_in": 20.0,
                          "m_dot_hot": mdot_arr, "m_dot_cold": mdot_arr,
                          "Rf_shell": 0.0002, "Rf_tube": 0.0002})

    print("=== EC073 Shell-and-Tube HX F1b — Simulation Summary ===")
    print("\n[Fouling sweep] T_h_in=90C, T_c_in=20C, m_dot=5 kg/s each:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  Rf={Rf_arr[i]*1000:.2f} m2mK/kW: Q={float(r_rf['Q_kw'][i]):.1f}kW, "
              f"eps={float(r_rf['effectiveness'][i]):.3f}, "
              f"CF={float(r_rf['cleanliness_factor'][i]):.3f}")

    print("\n[Flow sweep] Rf=0.0002 m2K/W each side:")
    for i in [0, 10, 25, 40, 49]:
        print(f"  m_dot={mdot_arr[i]:.1f} kg/s: Q={float(r_md['Q_kw'][i]):.1f}kW, "
              f"eps={float(r_md['effectiveness'][i]):.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Q vs Rf", "Effectiveness vs Rf",
                                            "Q vs Flow Rate"])
        fig.add_trace(go.Scatter(x=Rf_arr*1000, y=r_rf["Q_kw"], mode="lines",
                                 name="Q vs Rf"), row=1, col=1)
        fig.add_trace(go.Scatter(x=Rf_arr*1000, y=r_rf["effectiveness"], mode="lines",
                                 name="eps vs Rf"), row=1, col=2)
        fig.add_trace(go.Scatter(x=mdot_arr, y=r_md["Q_kw"], mode="lines",
                                 name="Q vs m_dot"), row=1, col=3)
        fig.update_layout(title="EC073 Shell-and-Tube HX — F1b Fouling", height=450)
        fig.update_xaxes(title_text="Rf [mm2K/W]", row=1, col=1)
        fig.update_xaxes(title_text="Rf [mm2K/W]", row=1, col=2)
        fig.update_xaxes(title_text="m_dot [kg/s]", row=1, col=3)
        fig.update_yaxes(title_text="Q [kW]", row=1, col=1)
        html_path = BASE / "simulation_report.html"
        fig.write_html(str(html_path))
        print(f"\nReport written to {html_path}")
    except ImportError:
        print("plotly not available — skipping HTML report")


if __name__ == "__main__":
    run()
