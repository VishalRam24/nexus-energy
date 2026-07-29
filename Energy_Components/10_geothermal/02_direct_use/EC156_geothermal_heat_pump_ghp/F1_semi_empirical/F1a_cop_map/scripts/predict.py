"""EC156 — Geothermal Heat Pump (GHP) — F1a COP Map — Standardized Predict Interface"""
import json
import numpy as np
from pathlib import Path
from model import GHPF1a


class ComponentModel:
    def __init__(self, params_path=None):
        base = Path(__file__).parent.parent
        if params_path is None:
            params_path = base / "data" / "parameters.json"
        with open(params_path) as f:
            self.params = json.load(f)
        self._model = GHPF1a(self.params)

    def predict(self, inputs: dict) -> dict:
        """
        Predict GHP (geothermal heat pump) performance.

        Parameters
        ----------
        inputs : dict
            T_source       : float or array  (degC, 0–25)  — ground loop fluid temperature
            T_sink         : float or array  (degC, 25–65) — heating load supply temperature
            part_load_ratio: float or array  (0–1, default=1) — thermal load fraction
            mode           : str ('heating' or 'cooling', default='heating')

        Returns
        -------
        dict
            cop_heating           : heating COP (-)
            cop_cooling           : cooling COP (-)
            heating_capacity_kw   : rated thermal output in heating mode (kW_th)
            electrical_input_kw   : compressor + aux electrical consumption (kW_e)
        """
        T_src = np.asarray(inputs["T_source"], dtype=float)
        T_snk = np.asarray(inputs["T_sink"],   dtype=float)
        plr   = np.asarray(inputs.get("part_load_ratio", 1.0), dtype=float)
        mode  = inputs.get("mode", "heating")

        cop_h = self._model.cop_heating(T_src, T_snk)
        cop_c = self._model.cop_cooling(T_src, T_snk)
        Q_h   = self._model.heating_capacity(T_src, T_snk, plr)

        if mode == "cooling":
            W_elec = self._model.electrical_input_cooling(T_src, T_snk, plr)
        else:
            W_elec = self._model.electrical_input_heating(T_src, T_snk, plr)

        return {
            "cop_heating": cop_h,
            "cop_cooling": cop_c,
            "heating_capacity_kw": Q_h,
            "electrical_input_kw": W_elec,
        }

    def get_info(self) -> dict:
        return {
            "name": "Geothermal Heat Pump (GHP)",
            "ec_id": "EC156",
            "fidelity": "F1a",
            "description": (
                "COP = eta_Carnot_frac * T_sink / (T_sink - T_source); "
                "ground-coupled, stable T_source=10-15°C gives COP advantage over ASHP"
            ),
            "inputs": {
                "T_source":        {"unit": "degC",  "range": [0.0,  25.0]},
                "T_sink":          {"unit": "degC",  "range": [25.0, 65.0]},
                "part_load_ratio": {"unit": "-",     "range": [0.0,  1.0], "default": 1.0},
                "mode":            {"unit": "str",   "options": ["heating", "cooling"], "default": "heating"},
            },
            "outputs": {
                "cop_heating":         {"unit": "dimensionless"},
                "cop_cooling":         {"unit": "dimensionless"},
                "heating_capacity_kw": {"unit": "kW_th"},
                "electrical_input_kw": {"unit": "kW_e"},
            },
            "source": "Staffell et al. (2012); ASHRAE (2011) Geothermal Heating and Cooling",
            "license": "BSD-3",
        }


if __name__ == "__main__":
    model = ComponentModel()
    # Standard rating: B0/W35 (ground 0°C, heating to 35°C)
    r = model.predict({"T_source": 10.0, "T_sink": 35.0})
    print(f"Ground T=10°C, T_load=35°C (typical residential heating):")
    print(f"  COP_heating = {float(r['cop_heating']):.2f}")
    print(f"  COP_cooling = {float(r['cop_cooling']):.2f}")
    print(f"  Q_heating   = {float(r['heating_capacity_kw']):.1f} kW_th")
    print(f"  W_elec      = {float(r['electrical_input_kw']):.2f} kW_e")

    # Comparison with ASHP at cold air temperature (showing GHP advantage)
    from model import GHPF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    ghp_model = GHPF1a(params)
    delta = ghp_model.cop_advantage_over_ashp(
        T_source_ghp_c=10.0, T_source_ashp_c=-5.0, T_sink_c=35.0
    )
    print(f"\n  GHP vs ASHP advantage at T_air=-5°C, T_gnd=10°C: +{float(delta):.2f} COP")
    info = model.get_info()
    print(f"  EC ID: {info['ec_id']}, Fidelity: {info['fidelity']}")
