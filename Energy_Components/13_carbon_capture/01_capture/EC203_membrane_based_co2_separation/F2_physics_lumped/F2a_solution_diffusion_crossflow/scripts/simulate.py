"""
EC203 -- Membrane-Based CO2 Separation -- F2a simulate / Plotly report.

Generates the purity-recovery tradeoff curve, area-resolved profiles, and the
effect of selectivity / pressure ratio. Plotly import is wrapped so its absence
does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1. Purity-recovery tradeoff vs area
    areas = np.logspace(0.3, 3.2, 40)
    recs, purs = [], []
    for A in areas:
        r = m.simulate(area_m2=float(A), n_eval=60)
        recs.append(r["recovery"])
        purs.append(r["purity"])

    # 2. Selectivity sweep
    sel_curves = {}
    for alpha in [10.0, 30.0, 50.0, 100.0]:
        mm = ComponentModel()._model
        mm.alpha = alpha
        mm.Q_N2 = mm.Q_CO2 / alpha
        rs, ps = [], []
        for A in areas:
            r = mm.simulate(area_m2=float(A), n_eval=40)
            rs.append(r["recovery"])
            ps.append(r["purity"])
        sel_curves[alpha] = (rs, ps)

    # 3. Area profile at a representative point
    prof = m.simulate(area_m2=100.0, n_eval=150)

    return areas, recs, purs, sel_curves, prof, m


def build_report(out_html="simulation_report.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    areas, recs, purs, sel_curves, prof, m = run_scenarios()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Purity-Recovery Tradeoff (selectivity sweep)",
            "Recovery & Purity vs Membrane Area",
            "Cross-flow Profiles along Module Area",
            "Retentate / Permeate composition vs Area",
        ),
    )

    for alpha, (rs, ps) in sel_curves.items():
        fig.add_trace(go.Scatter(x=np.array(rs) * 100, y=np.array(ps) * 100,
                                 mode="lines", name=f"alpha={alpha:.0f}"),
                      row=1, col=1)
    fig.update_xaxes(title_text="CO2 recovery [%]", row=1, col=1)
    fig.update_yaxes(title_text="permeate CO2 purity [%]", row=1, col=1)

    fig.add_trace(go.Scatter(x=areas, y=np.array(recs) * 100, name="recovery"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=areas, y=np.array(purs) * 100, name="purity"),
                  row=1, col=2)
    fig.update_xaxes(title_text="membrane area [m2]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="[%]", row=1, col=2)

    fig.add_trace(go.Scatter(x=prof["area"], y=prof["cum_permeate_CO2"],
                             name="cum permeate CO2 [mol/s]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=prof["area"], y=prof["F_CO2_retentate"],
                             name="retentate CO2 [mol/s]"), row=2, col=1)
    fig.update_xaxes(title_text="cumulative area [m2]", row=2, col=1)

    fig.add_trace(go.Scatter(x=prof["area"], y=prof["retentate_x_CO2"] * 100,
                             name="retentate x_CO2 [%]"), row=2, col=2)
    fig.add_trace(go.Scatter(x=prof["area"], y=prof["permeate_purity"] * 100,
                             name="cum permeate purity [%]"), row=2, col=2)
    fig.update_xaxes(title_text="cumulative area [m2]", row=2, col=2)

    fig.update_layout(
        title_text=f"EC203 Membrane CO2 Separation F2a — pressure ratio "
                   f"phi={m.pressure_ratio:.1f}, feed {m.y_feed*100:.0f}% CO2",
        height=820, width=1180,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(out_path)
    print(f"[simulate] Report written to {out_path}")
    return out_path


if __name__ == "__main__":
    build_report()
