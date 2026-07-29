"""EC133 — Tidal Lagoon — F1b Tidal Range / Efficiency — Standardized Predict Interface"""
import json, numpy as np
from pathlib import Path
from model import TidalLagoonF1b


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = TidalLagoonF1b(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        inputs:
            tidal_range_m   : float or array [m], peak-to-trough tidal range
            T_C             : float [degC], seawater temperature (optional, default T_ref)
            S_psu           : float [psu], salinity (optional, default S_ref)
            pumping_mode    : bool (optional, default False)
            lagoon_area_m2  : float (optional, default from params)
        returns:
            power_mw              : average electrical output [MW]
            turbine_efficiency    : eta at given head [-]
            energy_per_cycle_mwh  : energy per tidal period [MWh]
            seawater_density_kgm3 : rho(T,S) [kg/m3]
        """
        R     = np.asarray(inputs["tidal_range_m"], dtype=float)
        T_C   = inputs.get("T_C", None)
        S_psu = inputs.get("S_psu", None)
        pumping = bool(inputs.get("pumping_mode", False))
        area  = inputs.get("lagoon_area_m2", None)

        h = R / 2.0
        return {
            "power_mw":              self._model.avg_power_mw(R, area, T_C, S_psu, pumping),
            "turbine_efficiency":    self._model.turbine_efficiency(h),
            "energy_per_cycle_mwh":  self._model.energy_per_cycle_mwh(R, area, T_C, S_psu),
            "seawater_density_kgm3": self._model.seawater_density(T_C, S_psu),
        }

    def get_info(self) -> dict:
        m = self._model
        return {
            "name":     "Tidal Lagoon",
            "ec_id":    "EC133",
            "fidelity": "F1b",
            "model":    "Tidal Range / Head-Efficiency / Density Correction",
            "description": (
                f"Bidirectional tidal lagoon: efficiency vs head ratio "
                f"(eta_peak={m.eta_peak:.2f}, k_h={m.k_h:.2f}); "
                f"seawater density correction (T/S); "
                f"spring-neap amplitude ±{m.sn_amp*100:.0f}%; "
                f"optional pumping gain {m.pump_frac*100:.0f}%. "
                f"A_lagoon={m.A/1e6:.1f} km2, h_design={m.h_design:.1f} m."
            ),
            "inputs": {
                "tidal_range_m":  {"unit": "m",   "range": [0.0, 15.0]},
                "T_C":            {"unit": "degC","range": [0.0, 25.0], "optional": True},
                "S_psu":          {"unit": "psu", "range": [20.0, 40.0], "optional": True},
                "pumping_mode":   {"unit": "bool","optional": True},
                "lagoon_area_m2": {"unit": "m2",  "optional": True},
            },
            "outputs": {
                "power_mw":              {"unit": "MW"},
                "turbine_efficiency":    {"unit": "dimensionless"},
                "energy_per_cycle_mwh":  {"unit": "MWh"},
                "seawater_density_kgm3": {"unit": "kg/m3"},
            },
            "source": "Aggidis & Feather (2012) Ocean Eng. 40; Baker (1991) Tidal Power",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    r = model.predict({"tidal_range_m": 9.0, "T_C": 12.0, "S_psu": 35.0})
    print(f"Design range (9m): P={float(r['power_mw']):.1f} MW, "
          f"eta={float(r['turbine_efficiency']):.3f}, "
          f"E/cycle={float(r['energy_per_cycle_mwh']):.1f} MWh")
    r2 = model.predict({"tidal_range_m": 6.3, "T_C": 12.0, "S_psu": 35.0})
    print(f"Neap range (6.3m): P={float(r2['power_mw']):.1f} MW, "
          f"eta={float(r2['turbine_efficiency']):.3f}")
