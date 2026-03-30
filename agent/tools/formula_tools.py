"""Tools for formula prediction and verification."""

import math

import numpy as np
import torch

from mass_automation.formula import Formula, RealIsotopicDistribution
from mass_automation.formula.check_formula import check_presence
from mass_automation.formula.determination import brute_force_search
from mass_automation.utils import ELEMENT_DICT


PREDICT_FORMULA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_formula",
        "description": (
            "Predict the elemental composition of an isotopic cluster using the MLP neural network. "
            "Requires deisotoping to have been run first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "integer",
                    "description": "The cluster ID to predict formula for.",
                },
            },
            "required": ["cluster_id"],
        },
    },
}

CHECK_FORMULA_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_formula",
        "description": (
            "Verify whether a molecular formula is present in the loaded spectrum. "
            "Compares theoretical isotopic distribution against experimental peaks. "
            "Lower cosine distance = better match (< 0.01 is good, < 0.001 is excellent)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "formula_string": {
                    "type": "string",
                    "description": "Molecular formula string, e.g. 'C10H15NO2' or 'C27H36N2PdCl'.",
                },
                "charge": {
                    "type": "integer",
                    "description": "Charge state of the ion (default: 1).",
                },
                "cal_error": {
                    "type": "number",
                    "description": "Calibration error radius for first peak search (default: 0.006).",
                },
                "dist_error": {
                    "type": "number",
                    "description": "Distance error for subsequent peaks (default: 0.003).",
                },
            },
            "required": ["formula_string"],
        },
    },
}

BRUTE_FORCE_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "brute_force_search",
        "description": (
            "Enumerate all molecular formulas matching an exact mass within tolerance, "
            "then verify each against the spectrum. Can be slow for large search spaces."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "exact_mass": {
                    "type": "number",
                    "description": "The exact mass to search for.",
                },
                "elements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Elements to consider, e.g. ['C', 'H', 'N', 'O'].",
                },
                "low_limits": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Minimum count for each element.",
                },
                "high_limits": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Maximum count for each element.",
                },
                "mass_error": {
                    "type": "number",
                    "description": "Mass tolerance for enumeration (default: 0.005).",
                },
                "cosine_threshold": {
                    "type": "number",
                    "description": "Max cosine distance to include in results (default: 0.1).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20).",
                },
            },
            "required": ["exact_mass", "elements", "low_limits", "high_limits"],
        },
    },
}


def _safe_float(v):
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return f


def predict_formula(state, registry, cluster_id: int) -> dict:
    if state.spectrum is None:
        return {"error": "No spectrum loaded."}
    if state.labels is None:
        return {"error": "No deisotoping results. Run deisotope first."}
    if registry.mlp_model is None:
        return {"error": "MLP model not loaded."}

    try:
        indices = state.get_cluster_indices(cluster_id)
    except ValueError as e:
        return {"error": str(e)}

    rid = RealIsotopicDistribution(state.spectrum, indices)
    representations = rid.get_representation(length=registry.mlp_vector_length + 1)

    vectors = [rep[0] for rep in representations]
    input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32)

    with torch.no_grad():
        output = registry.mlp_model(input_tensor)

    avg_output = output.mean(dim=0).numpy()

    elements = []
    for idx in range(len(avg_output)):
        count = float(avg_output[idx])
        element_number = idx + 1
        if count > 0.5 and element_number in ELEMENT_DICT:
            elements.append({
                "symbol": ELEMENT_DICT[element_number],
                "count": round(count, 2),
            })

    # Build a formula string from rounded counts
    formula_str = ""
    for el in sorted(elements, key=lambda e: e["symbol"]):
        n = round(el["count"])
        if n > 0:
            formula_str += f"{el['symbol']}{n}" if n > 1 else el["symbol"]

    return {
        "cluster_id": cluster_id,
        "predicted_elements": elements,
        "suggested_formula": formula_str,
    }


def check_formula(state, registry, formula_string: str, charge: int = 1,
                   cal_error: float = None, dist_error: float = None) -> dict:
    if state.spectrum is None:
        return {"error": "No spectrum loaded."}

    try:
        formula = Formula(formula_string, charge=charge)
    except Exception as e:
        return {"error": f"Invalid formula '{formula_string}': {e}"}

    kwargs = {}
    if cal_error is not None:
        kwargs["cal_error"] = cal_error
    if dist_error is not None:
        kwargs["dist_error"] = dist_error

    try:
        cos_dist, real_coords, matched_pct, mass_error = check_presence(
            state.spectrum, formula, **kwargs
        )
    except Exception as e:
        return {"error": f"check_presence failed: {e}"}

    return {
        "formula": formula_string,
        "charge": charge,
        "cosine_distance": _safe_float(cos_dist),
        "matched_percentage": _safe_float(matched_pct),
        "mass_error_ppm": _safe_float(mass_error),
        "matched_masses": [_safe_float(m) for m in real_coords[0]],
        "matched_intensities": [_safe_float(i) for i in real_coords[1]],
    }


def brute_force_search_tool(state, registry, exact_mass: float, elements: list,
                             low_limits: list, high_limits: list,
                             mass_error: float = 0.005, cosine_threshold: float = 0.1,
                             max_results: int = 20) -> dict:
    if state.spectrum is None:
        return {"error": "No spectrum loaded."}

    check_params = {"cal_error": 0.006, "dist_error": 0.003}

    results = []
    for item in brute_force_search(
        exact_mass, elements, low_limits, high_limits,
        state.spectrum, mass_error, check_params, cosine_threshold,
    ):
        formula_counts, mass, cosine_dist, ppm = item
        formula_str = "".join(
            f"{el}{int(c)}" if c > 1 else el
            for el, c in zip(elements, formula_counts) if c > 0
        )
        results.append({
            "formula": formula_str,
            "mass": round(float(mass), 6),
            "cosine_distance": _safe_float(cosine_dist),
            "mass_error_ppm": _safe_float(ppm),
        })
        if len(results) >= max_results:
            break

    results.sort(key=lambda r: r["cosine_distance"])

    return {
        "exact_mass": exact_mass,
        "num_results": len(results),
        "candidates": results,
    }
