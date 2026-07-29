"""EC197 F1a — DME Synthesis simulation scenarios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    T = np.linspace(200, 340, 30)
    r = m.predict({"n_co_in": 1.0, "temperature_C": T, "pressure_bar": 40.0})
    print("EC197 F1a — DME Synthesis Reactor")
    print(f"  CO conversion at 260C, 40 bar: {float(m.model.conversion(260, 40)):.4f}")
    print(f"  DME selectivity at 265C: {float(m.model.selectivity_dme(265)):.4f}")
    print(f"  DME production (n_CO=1 mol/s): {float(m.model.dme_production_mol_s(260, 40)):.4f} mol/s")
    print(f"  Energy efficiency at design: {float(m.model.energy_efficiency(260, 40)):.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
