"""Optional Plotly report for EC198 Post-Combustion Capture (Amine Scrubbing) F0a (safe if plotly missing)."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    print("Sample:", m.predict({'flue_gas_rate': 500.0, 'co2_fraction': 0.12, 'capture_rate': 0.9}))
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report.")
        return
    print("plotly available -- report stub.")


if __name__ == "__main__":
    main()
