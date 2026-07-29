"""Optional Plotly report for EC199 Pre-Combustion Capture (WGS + Separation) F0a (safe if plotly missing)."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    print("Sample:", m.predict({'T_WGS_C': 250.0, 'co_flow_kg_s': 10.0}))
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report.")
        return
    print("plotly available -- report stub.")


if __name__ == "__main__":
    main()
