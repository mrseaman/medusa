"""Tools for loading and manipulating spectra."""

import os
import pickle as pkl

import numpy as np

from mass_automation.experiment import Spectrum, Experiment


LOAD_SPECTRUM_SCHEMA = {
    "type": "function",
    "function": {
        "name": "load_spectrum",
        "description": (
            "Load a mass spectrum from a .pkl or .mzXML file. "
            "For mzXML files, all scans are summarized into a single spectrum."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the spectrum file (.pkl or .mzXML).",
                },
            },
            "required": ["file_path"],
        },
    },
}

GET_SPECTRUM_SLICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_spectrum_slice",
        "description": "Extract a sub-spectrum within a specific mass range.",
        "parameters": {
            "type": "object",
            "properties": {
                "left_mass": {
                    "type": "number",
                    "description": "Left bound of the mass range.",
                },
                "right_mass": {
                    "type": "number",
                    "description": "Right bound of the mass range.",
                },
            },
            "required": ["left_mass", "right_mass"],
        },
    },
}


def load_spectrum(state, registry, file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pkl":
        with open(file_path, "rb") as f:
            obj = pkl.load(f)
        if isinstance(obj, Spectrum):
            state.spectrum = obj
        elif isinstance(obj, (tuple, list)) and len(obj) >= 2:
            state.spectrum = Spectrum(masses=np.array(obj[0]), ints=np.array(obj[1]))
        else:
            return {"error": f"Unexpected pickle content: {type(obj).__name__}"}
        state.experiment = None
    elif ext in (".mzxml", ".mzml"):
        exp = Experiment(file_path, verbose=False)
        state.experiment = exp
        state.spectrum = exp.summarize()
    else:
        return {"error": f"Unsupported format: {ext}"}

    state.source_path = file_path
    state.labels = None

    spec = state.spectrum
    return {
        "status": "loaded",
        "file": file_path,
        "num_points": len(spec.masses),
        "mass_range": [round(float(spec.masses.min()), 4), round(float(spec.masses.max()), 4)],
        "max_intensity": round(float(spec.ints.max()), 2),
    }


def get_spectrum_slice(state, registry, left_mass: float, right_mass: float) -> dict:
    if state.spectrum is None:
        return {"error": "No spectrum loaded. Use load_spectrum first."}

    state.spectrum = state.spectrum.get_slice(left_mass, right_mass)
    state.labels = None

    spec = state.spectrum
    return {
        "status": "sliced",
        "mass_range": [round(float(spec.masses.min()), 4), round(float(spec.masses.max()), 4)],
        "num_points": len(spec.masses),
    }
