"""Tools for deisotoping spectra."""

import numpy as np


DEISOTOPE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "deisotope",
        "description": (
            "Run ML-based deisotoping on the loaded spectrum. "
            "Groups peaks into isotopic clusters. Returns cluster summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["xgb", "rf", "cb"],
                    "description": "Deisotoping model to use (default: xgb).",
                },
                "threshold": {
                    "type": "number",
                    "description": "Prediction probability threshold (default: 0.5).",
                },
            },
        },
    },
}

LIST_CLUSTERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_clusters",
        "description": (
            "List all isotopic clusters from the last deisotoping run. "
            "Shows cluster ID, monoisotopic mass, and number of peaks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "enum": ["mass", "intensity", "size"],
                    "description": "Sort clusters by mass, peak intensity, or number of peaks (default: mass).",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Return only the top N clusters (default: all).",
                },
            },
        },
    },
}

EXTRACT_CLUSTER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_cluster",
        "description": "Get the peaks (masses and intensities) for a specific isotopic cluster.",
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "integer",
                    "description": "The cluster ID to extract.",
                },
            },
            "required": ["cluster_id"],
        },
    },
}


def deisotope(state, registry, model: str = "xgb", threshold: float = 0.5) -> dict:
    if state.spectrum is None:
        return {"error": "No spectrum loaded. Use load_spectrum first."}

    deisotoper = registry.get_deisotoper(model)
    labels = deisotoper.run(state.spectrum, threshold=threshold)
    state.labels = labels

    summary = state.get_cluster_summary()
    return {
        "status": "deisotoped",
        "model": model,
        "threshold": threshold,
        "num_clusters": len(summary),
        "noise_peaks": int((labels == -1).sum()),
        "clusters_preview": [
            {"id": cid, "monoisotopic_mass": round(m, 4), "num_peaks": n}
            for cid, m, n in summary[:20]
        ],
    }


def list_clusters(state, registry, sort_by: str = "mass", top_n: int = None) -> dict:
    if state.labels is None:
        return {"error": "No deisotoping results. Run deisotope first."}

    summary = state.get_cluster_summary()

    if sort_by == "intensity":
        summary.sort(key=lambda x: -state.spectrum.ints[
            np.where(state.labels == x[0])[0]
        ].max())
    elif sort_by == "size":
        summary.sort(key=lambda x: -x[2])
    # default is mass order (already sorted)

    if top_n:
        summary = summary[:top_n]

    clusters = []
    for cid, mono_mass, n_peaks in summary:
        idx = np.where(state.labels == cid)[0]
        max_int = float(state.spectrum.ints[idx].max())
        clusters.append({
            "id": cid,
            "monoisotopic_mass": round(mono_mass, 4),
            "num_peaks": n_peaks,
            "max_intensity": round(max_int, 2),
        })

    return {"num_clusters": len(clusters), "clusters": clusters}


def extract_cluster(state, registry, cluster_id: int) -> dict:
    if state.labels is None:
        return {"error": "No deisotoping results. Run deisotope first."}

    try:
        indices = state.get_cluster_indices(cluster_id)
    except ValueError as e:
        return {"error": str(e)}

    masses = state.spectrum.masses[indices]
    ints = state.spectrum.ints[indices]

    peaks = [
        {"mass": round(float(m), 6), "intensity": round(float(i), 2)}
        for m, i in zip(masses, ints)
    ]

    return {
        "cluster_id": cluster_id,
        "monoisotopic_mass": round(float(masses[0]), 6),
        "num_peaks": len(peaks),
        "peaks": peaks,
    }
