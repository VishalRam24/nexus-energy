"""EC012 — Compressed Gas H2 Storage — F1b Real-Gas — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import CompressedGasH2F1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = CompressedGasH2F1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            P_bar       : Tank pressure [bar]  (float or array)
            T_K         : Tank gas temperature [K]  (float or array)
            T_amb_K     : Ambient temperature [K]  (optional, default 298.15)
            mode        : 'storage' | 'fill' | 'compression' (default 'storage')
            P1_bar      : Inlet pressure for compression [bar]  (mode='compression')
            P2_bar      : Outlet pressure for compression [bar] (mode='compression')
        returns (mode='storage'):
            stored_mass_kg, energy_stored_MJ, fill_fraction, Z,
            gravimetric_density_wt_pct, volumetric_density_kg_per_m3
        returns (mode='fill'):
            T_post_fill_K, dT_K, tau_s, stored_mass_kg, Z
        returns (mode='compression'):
            compression_work_kJ_per_kg, Z_inlet
        """
        mode = inputs.get("mode", "storage")
        T_amb = float(inputs.get("T_amb_K", self._model.T_amb_default))

        if mode == "compression":
            P1 = np.asarray(inputs["P1_bar"], dtype=float)
            P2 = np.asarray(inputs["P2_bar"], dtype=float)
            T1 = float(inputs.get("T1_K", self._model.T_inlet_default))
            return {
                "compression_work_kJ_per_kg": self._model.compression_work(P1, P2, T1),
                "Z_inlet": self._model.compressibility_factor(P1, T1),
            }

        if mode == "fill":
            P1 = np.asarray(inputs["P1_bar"], dtype=float)
            P2 = np.asarray(inputs["P2_bar"], dtype=float)
            T_post = self._model.tank_temperature_after_fill(P1, P2, T_amb)
            dT = T_post - T_amb
            m2 = self._model.stored_mass(P2, T_post)
            Z2 = self._model.compressibility_factor(P2, T_post)
            return {
                "T_post_fill_K": T_post,
                "dT_K": dT,
                "tau_s": self._model.thermal_equilibration_time(),
                "stored_mass_kg": m2,
                "Z": Z2,
            }

        # mode == 'storage'
        P = np.asarray(inputs["P_bar"], dtype=float)
        T = np.asarray(inputs.get("T_K", T_amb), dtype=float)
        return {
            "stored_mass_kg": self._model.stored_mass(P, T),
            "energy_stored_MJ": self._model.energy_stored(P, T),
            "fill_fraction": self._model.fill_fraction(P, T),
            "Z": self._model.compressibility_factor(P, T),
            "gravimetric_density_wt_pct": self._model.gravimetric_density(P, T),
            "volumetric_density_kg_per_m3": self._model.volumetric_density(P, T),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name": "Compressed Gas H2 Storage (Real-Gas + Thermal)",
            "ec_id": "EC012",
            "fidelity": "F1b",
            "description": (
                "Real-gas Z(T,P) via truncated virial equation; "
                "ambient T coupling; heat-of-compression transient during fill; "
                "real-gas corrected compression work."
            ),
            "inputs": {
                "P_bar": {"unit": "bar", "range": [1.0, 900.0]},
                "T_K": {"unit": "K", "range": [233.0, 373.0]},
                "T_amb_K": {"unit": "K", "range": [233.0, 333.0], "default": 298.15},
                "mode": {"values": ["storage", "fill", "compression"]},
            },
            "outputs": {
                "stored_mass_kg": {"unit": "kg"},
                "energy_stored_MJ": {"unit": "MJ"},
                "fill_fraction": {"unit": "dimensionless"},
                "Z": {"unit": "dimensionless"},
                "gravimetric_density_wt_pct": {"unit": "wt%"},
                "volumetric_density_kg_per_m3": {"unit": "kg/m3"},
                "compression_work_kJ_per_kg": {"unit": "kJ/kg"},
                "T_post_fill_K": {"unit": "K"},
                "dT_K": {"unit": "K"},
                "tau_s": {"unit": "s"},
            },
            "params": {
                "V_tank": f"{m.V_tank} m3",
                "P_max": f"{m.P_max} bar",
                "P_min": f"{m.P_min} bar",
                "T_amb_default": f"{m.T_amb_default} K",
                "eta_isentropic": m.eta_s,
            },
            "source": "Leachman et al. (2009); Zheng et al. (2012); Lemmon et al. (2008)",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    print("=== EC012 F1b Real-Gas H2 Storage ===\n")
    print("Z(T,P) at 300 K:")
    for P in [1, 10, 100, 200, 350, 700]:
        r = model.predict({"P_bar": float(P), "T_K": 300.0})
        print(f"  P={P:>4d} bar  Z={float(r['Z']):.4f}  "
              f"m={float(r['stored_mass_kg']):.3f} kg  "
              f"fill={float(r['fill_fraction']):.3f}")

    print("\nCompression work (30→700 bar, 298 K):")
    r = model.predict({"mode": "compression", "P1_bar": 30.0, "P2_bar": 700.0})
    print(f"  w_comp = {float(r['compression_work_kJ_per_kg']):.1f} kJ/kg  "
          f"Z_inlet = {float(r['Z_inlet']):.4f}")

    print("\nFill transient (20→700 bar):")
    r = model.predict({"mode": "fill", "P1_bar": 20.0, "P2_bar": 700.0, "T_amb_K": 298.15})
    print(f"  T_post_fill = {float(r['T_post_fill_K']):.1f} K  "
          f"dT = {float(r['dT_K']):.1f} K  tau = {float(r['tau_s']):.0f} s")
