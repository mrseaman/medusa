"""Tool registry — collects all tool schemas and dispatch functions."""

from .spectrum_tools import (
    LOAD_SPECTRUM_SCHEMA, load_spectrum,
    GET_SPECTRUM_SLICE_SCHEMA, get_spectrum_slice,
)
from .deisotoping_tools import (
    DEISOTOPE_SCHEMA, deisotope,
    LIST_CLUSTERS_SCHEMA, list_clusters,
    EXTRACT_CLUSTER_SCHEMA, extract_cluster,
)
from .formula_tools import (
    PREDICT_FORMULA_SCHEMA, predict_formula,
    CHECK_FORMULA_SCHEMA, check_formula,
    BRUTE_FORCE_SEARCH_SCHEMA, brute_force_search_tool,
)

# Maps function name -> (schema, callable)
TOOL_REGISTRY = {
    "load_spectrum": (LOAD_SPECTRUM_SCHEMA, load_spectrum),
    "get_spectrum_slice": (GET_SPECTRUM_SLICE_SCHEMA, get_spectrum_slice),
    "deisotope": (DEISOTOPE_SCHEMA, deisotope),
    "list_clusters": (LIST_CLUSTERS_SCHEMA, list_clusters),
    "extract_cluster": (EXTRACT_CLUSTER_SCHEMA, extract_cluster),
    "predict_formula": (PREDICT_FORMULA_SCHEMA, predict_formula),
    "check_formula": (CHECK_FORMULA_SCHEMA, check_formula),
    "brute_force_search": (BRUTE_FORCE_SEARCH_SCHEMA, brute_force_search_tool),
}


def get_tool_schemas():
    """Return the list of tool schemas for the OpenAI API."""
    return [schema for schema, _ in TOOL_REGISTRY.values()]


def dispatch(name: str, state, registry, **kwargs):
    """Call a tool by name with the given arguments."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    _, func = TOOL_REGISTRY[name]
    return func(state, registry, **kwargs)
