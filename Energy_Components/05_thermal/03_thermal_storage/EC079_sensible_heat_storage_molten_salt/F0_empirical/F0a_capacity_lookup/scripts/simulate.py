"""OPTIONAL F0a simulation report for Molten Salt TES (EC079). Plotly wrapped in try/except."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    m = ComponentModel()
    try:
        import plotly.graph_objects as go  # noqa: F401
    except Exception:
        print("plotly not available; skipping HTML report. Model loads OK:",
              m.component_id, m.fidelity)
        return
    print("Report generation stub for", m.component_id)


if __name__ == "__main__":
    main()
