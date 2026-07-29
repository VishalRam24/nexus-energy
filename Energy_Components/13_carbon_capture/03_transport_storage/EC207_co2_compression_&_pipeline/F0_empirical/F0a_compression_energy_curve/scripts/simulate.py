"""Optional Plotly report for EC207 CO2 Compression & Pipeline F0a (safe if plotly missing)."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    print("Sample:", m.predict({'mass_flow': 100.0, 'P_outlet': 150.0, 'pipeline_length_km': 100.0}))
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report.")
        return
    print("plotly available -- report stub.")


if __name__ == "__main__":
    main()
