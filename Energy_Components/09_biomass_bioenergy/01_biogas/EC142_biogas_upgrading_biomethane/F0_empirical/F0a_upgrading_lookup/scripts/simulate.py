"""Optional Plotly report for EC142 F0a upgrading lookup."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    raws = list(range(50, 71))
    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping HTML report")
        for fs in m.table.feedstocks:
            print(f"{fs:16s} {m.predict({'feedstock': fs})['biomethane_per_biogas']:.3f}")
        return
    ys = [m.predict({"raw_CH4_pct": c})["biomethane_per_biogas"] for c in raws]
    fig = go.Figure(go.Scatter(x=raws, y=ys, mode="lines+markers"))
    fig.update_layout(title="EC142 F0a biomethane yield vs raw CH4",
                      xaxis_title="Raw biogas CH4 (%)", yaxis_title="Nm3 biomethane / Nm3 biogas")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
