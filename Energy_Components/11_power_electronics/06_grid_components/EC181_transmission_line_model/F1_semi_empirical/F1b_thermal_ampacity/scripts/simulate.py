"""EC181 F1b — Thermal Ampacity simulation scenarios."""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()

    # Scenario 1: R(T) sensitivity
    T_range = np.linspace(-10, 75, 30)
    r_ac = [m.model.r_ac_pu_per_km(T) * 200.0 for T in T_range]

    # Scenario 2: Ampacity vs ambient temperature
    T_amb = np.linspace(-20, 50, 30)
    I_max = [m.model.thermal_ampacity_A(T_amb_C=T) for T in T_amb]

    # Scenario 3: Load sweep at two temperatures
    P = np.linspace(0.05, 1.2, 30)
    r_hot  = m.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                         "P_load_pu": P, "Q_load_pu": P*0.3,
                         "T_cond_C": 75.0, "T_amb_C": 40.0})
    r_cold = m.predict({"V_s_pu": 1.0, "delta_s_rad": 0.0,
                         "P_load_pu": P, "Q_load_pu": P*0.3,
                         "T_cond_C": 20.0, "T_amb_C": 10.0})

    print("EC181 F1b — Thermal Ampacity Simulation")
    print(f"  R_ac at 20 C: {m.model.r_ac_pu_per_km(20)*200:.6f} pu")
    print(f"  R_ac at 75 C: {m.model.r_ac_pu_per_km(75)*200:.6f} pu")
    print(f"  Ampacity at 25 degC ambient: {m.model.thermal_ampacity_A(25.0):.1f} A")
    print(f"  Ampacity at 40 degC ambient: {m.model.thermal_ampacity_A(40.0):.1f} A")
    print(f"  Derating at 40 degC: {m.model.ampacity_derating_factor(40.0):.4f}")
    print(f"  P_loss hot (P=0.5 pu): {float(r_hot['P_loss_pu'][14]):.5f} pu")
    print(f"  P_loss cold (P=0.5 pu): {float(r_cold['P_loss_pu'][14]):.5f} pu")
    print("DONE")


if __name__ == "__main__":
    main()
