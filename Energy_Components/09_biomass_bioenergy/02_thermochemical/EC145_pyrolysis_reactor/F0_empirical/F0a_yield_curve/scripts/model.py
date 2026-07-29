"""F0a empirical product-yield curves for EC145 pyrolysis reactor.

Black-box model: bio-oil / char / gas mass-fraction yields vs reactor
temperature (np.interp over breakpoints). Bio-oil peaks at ~500 C. Fractions
are renormalised to sum to 1.0. NumPy only.

Data source: Bridgwater (2012) — yields reused from the EC145 F1b parameter set.
"""
import numpy as np


class YieldCurve:
    def __init__(self, params):
        r = params["rated"]
        self.T_peak = r["T_bio_oil_peak_degC"]["value"]
        self.lhv = {"bio_oil": r["LHV_bio_oil_MJ_kg"]["value"],
                    "char": r["LHV_char_MJ_kg"]["value"],
                    "gas": r["LHV_gas_MJ_kg"]["value"]}
        yc = params["yield_curves"]
        self.T_bp = np.asarray(yc["T_degC"], float)
        self.oil_bp = np.asarray(yc["bio_oil_frac"], float)
        self.char_bp = np.asarray(yc["char_frac"], float)
        self.gas_bp = np.asarray(yc["gas_frac"], float)

    def yields(self, T_degC):
        """Return normalised {bio_oil, char, gas} mass fractions at temperature."""
        oil = float(np.interp(T_degC, self.T_bp, self.oil_bp))
        char = float(np.interp(T_degC, self.T_bp, self.char_bp))
        gas = float(np.interp(T_degC, self.T_bp, self.gas_bp))
        tot = oil + char + gas
        return {"bio_oil": oil / tot, "char": char / tot, "gas": gas / tot}

    def energy_density_MJ_kg(self, T_degC):
        """LHV-weighted energy content of products per kg feed."""
        y = self.yields(T_degC)
        return sum(y[k] * self.lhv[k] for k in y)
