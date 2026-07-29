"""EC188 F1b — SMES self-discharge and cryo load simulation scenarios."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def main():
    m = ComponentModel()
    print("EC188 F1b — SMES Self-Discharge + Cryogenic Load")
    print(f"  E_max = {m.model.E_max_MJ:.2f} MJ")
    print(f"  I_max = {m.model.I_max:.1f} A")
    print(f"  P_cryo @ 20K, idle: {m.model.cryo_power_MW(20.0, 0.0)*1e3:.2f} kW")
    print(f"  P_cryo @ 30K: {m.model.cryo_power_MW(30.0, 0.0)*1e3:.2f} kW")
    print(f"  Self-discharge tau @ SOC=1.0: {m.model.self_discharge_tau_h(1.0):.1f} h")
    print(f"  Self-discharge tau @ SOC=0.5: {m.model.self_discharge_tau_h(0.5):.1f} h")

    r_dis = m.predict({"SOC": 0.8, "P_request_MW": 1.5, "mode": "discharge"})
    r_chg = m.predict({"SOC": 0.3, "P_request_MW": 1.5, "mode": "charge"})
    print(f"  Discharge: eta_inst={float(r_dis['eta_instantaneous']):.4f}, "
          f"P_cryo={float(r_dis['P_cryo_load_MW'])*1e3:.2f} kW")
    print(f"  Charge:    eta_inst={float(r_chg['eta_instantaneous']):.4f}, "
          f"P_cryo={float(r_chg['P_cryo_load_MW'])*1e3:.2f} kW")
    print("DONE")


if __name__ == "__main__":
    main()
