"""EC140 -- Anaerobic Digester -- F1b Feedstock -- Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import AnaerobicDigesterF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    ad = AnaerobicDigesterF1b(params)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Methane Yield by Feedstock",
            "Methane Yield vs HRT",
            "Temperature Effect on Yield",
            "C/N Ratio by Feedstock",
            "Co-Digestion Synergy",
            "VS Loading Effect",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    feedstocks = list(ad.feedstock_db.keys())
    colors = ["firebrick", "steelblue", "green", "purple", "darkorange"]

    # --- 1) Bar chart: BMP by feedstock ---
    bmps = [ad.feedstock_db[fs]["BMP_L_CH4_kgVS"] for fs in feedstocks]
    fig.add_trace(
        go.Bar(x=feedstocks, y=bmps, name="BMP", marker_color=colors),
        row=1, col=1,
    )

    # --- 2) Methane yield vs HRT for each feedstock ---
    hrt_range = np.linspace(5, 60, 50)
    for i, fs in enumerate(feedstocks):
        yields = []
        for hrt in hrt_range:
            r = ad.predict(fs, 3.0, 37.0, float(hrt))
            yields.append(r["methane_yield_m3_day"])
        fig.add_trace(
            go.Scatter(x=hrt_range, y=yields, name=fs,
                       line=dict(color=colors[i], width=2)),
            row=1, col=2,
        )

    # --- 3) Temperature effect ---
    T_range = np.linspace(25, 45, 50)
    for i, fs in enumerate(["food_waste", "cattle_manure"]):
        yields = []
        for T in T_range:
            r = ad.predict(fs, 3.0, float(T), 20.0)
            yields.append(r["methane_yield_m3_day"])
        fig.add_trace(
            go.Scatter(x=T_range, y=yields, name=fs,
                       line=dict(width=2)),
            row=1, col=3,
        )

    # --- 4) C/N ratios ---
    cn_ratios = []
    for fs in feedstocks:
        blend = ad.blend_properties(fs)
        cn_ratios.append(blend["cn_ratio"])
    fig.add_trace(
        go.Bar(x=feedstocks, y=cn_ratios, name="C/N ratio", marker_color=colors),
        row=2, col=1,
    )
    fig.add_hline(y=20, line_dash="dash", line_color="green",
                  annotation_text="Optimal min", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  annotation_text="Optimal max", row=2, col=1)

    # --- 5) Co-digestion: manure + food waste at various ratios ---
    fractions = np.linspace(0, 1, 20)
    yields_blend = []
    cn_blend = []
    for f in fractions:
        if f == 0:
            blend = {"food_waste": 1.0}
        elif f == 1:
            blend = {"cattle_manure": 1.0}
        else:
            blend = {"cattle_manure": float(f), "food_waste": float(1 - f)}
        r = ad.predict(blend, 3.0, 37.0, 20.0)
        yields_blend.append(r["methane_yield_m3_day"])
        cn_blend.append(r["cn_ratio"])

    fig.add_trace(
        go.Scatter(x=fractions * 100, y=yields_blend, name="CH4 yield (blend)",
                   line=dict(color="firebrick", width=2)),
        row=2, col=2,
    )

    # --- 6) VS loading effect ---
    vs_range = np.linspace(0.5, 8, 30)
    for fs in ["food_waste", "cattle_manure", "corn_silage"]:
        yields_vs = []
        for vs in vs_range:
            r = ad.predict(fs, float(vs), 37.0, 20.0)
            yields_vs.append(r["methane_yield_m3_day"])
        fig.add_trace(
            go.Scatter(x=vs_range, y=yields_vs, name=fs, line=dict(width=2)),
            row=2, col=3,
        )

    # Axes
    fig.update_xaxes(title_text="Feedstock", row=1, col=1)
    fig.update_xaxes(title_text="HRT (days)", row=1, col=2)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=3)
    fig.update_xaxes(title_text="Feedstock", row=2, col=1)
    fig.update_xaxes(title_text="Cattle manure fraction (%)", row=2, col=2)
    fig.update_xaxes(title_text="VS loading (kgVS/m3/day)", row=2, col=3)
    fig.update_yaxes(title_text="BMP (L CH4/kgVS)", row=1, col=1)
    fig.update_yaxes(title_text="CH4 yield (m3/day)", row=1, col=2)
    fig.update_yaxes(title_text="CH4 yield (m3/day)", row=1, col=3)
    fig.update_yaxes(title_text="C/N ratio", row=2, col=1)
    fig.update_yaxes(title_text="CH4 yield (m3/day)", row=2, col=2)
    fig.update_yaxes(title_text="CH4 yield (m3/day)", row=2, col=3)

    fig.update_layout(
        title=(
            f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Feedstock Variation<br>"
            f"<sup>5 feedstocks | Co-digestion synergy | C/N optimization | 2000 m3</sup>"
        ),
        height=900,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to: {out}")


if __name__ == "__main__":
    generate_report()
