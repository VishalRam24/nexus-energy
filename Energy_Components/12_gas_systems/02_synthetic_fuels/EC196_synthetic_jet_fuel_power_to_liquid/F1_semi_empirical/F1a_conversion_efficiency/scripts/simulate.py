"""EC196 F1a — Synthetic Jet Fuel FT Conversion simulation scenarios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    T = np.linspace(180, 280, 30)
    r = m.predict({"n_co_in": 1.0, "temperature_C": T, "pressure_bar": 25.0})
    print("EC196 F1a — FT Jet Fuel Conversion")
    print(f"  ASF C8-C16 selectivity (alpha={m.model.alpha_ASF}): {m.model.asf_selectivity_jet():.4f}")
    print(f"  CO conversion at 220C, 25 bar: {float(m.model.conversion(220, 25)):.4f}")
    print(f"  Energy efficiency at design: {float(m.model.energy_efficiency(220, 25)):.4f}")
    print(f"  Jet fuel yield at design (n_CO=1 mol/s): {float(m.model.jet_fuel_yield_mol_s(220, 25)):.4f} mol/s")
    print("DONE")


if __name__ == "__main__":
    main()
