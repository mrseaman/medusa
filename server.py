import logging
import math
import sys
import types
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# pyopenms is only needed for file I/O (not used by the API).
# Provide a stub if it's not installed so mass_automation can still import.
if "pyopenms" not in sys.modules:
    try:
        import pyopenms  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("pyopenms")
        _stub.MSExperiment = None
        _stub.MzXMLFile = None
        sys.modules["pyopenms"] = _stub

from mass_automation.deisotoping.process import MlDeisotoper
from mass_automation.experiment import Spectrum
from mass_automation.formula import Formula, RealIsotopicDistribution
from mass_automation.formula.check_formula import check_presence
from mass_automation.formula.model import MLP
from mass_automation.utils import ELEMENT_DICT

logger = logging.getLogger(__name__)

app = FastAPI(title="MEDUSA", description="Mass spectrometry analysis API")

# --- Model singletons (loaded at startup) ---

deisotoping_models: dict = {}
mlp_model: MLP = None
mlp_vector_length: int = 500  # will be set from checkpoint hparams


@app.on_event("startup")
def load_models():
    global deisotoping_models, mlp_model, mlp_vector_length

    logger.info("Loading deisotoping models...")
    for name, path in [
        ("xgb", "data/models/xgb_model.pkl"),
        ("rf", "data/models/rf_model.pkl"),
        ("cb", "data/models/cb_model.pkl"),
    ]:
        try:
            deisotoping_models[name] = MlDeisotoper().load(path)
            logger.info(f"  Loaded {name} from {path}")
        except Exception as e:
            logger.warning(f"  Could not load {name}: {e}")

    logger.info("Loading MLP model...")
    try:
        mlp_model = MLP.load_from_checkpoint(
            "nn_models/regression/mlp_0.25.ckpt",
            map_location="cpu",
        )
        mlp_model.eval()
        logger.info("  MLP loaded")
    except Exception:
        # load_from_checkpoint can fail due to torchmetrics incompatibility,
        # weights_only default changes, etc. Fall back to manual loading.
        logger.info("  Retrying MLP load (inference-only fallback)...")
        checkpoint = torch.load(
            "nn_models/regression/mlp_0.25.ckpt",
            map_location="cpu",
            weights_only=False,
        )
        from mass_automation.formula.model import LinearWithHidden
        from mass_automation.utils import Element

        hparams = checkpoint.get("hyper_parameters", {})
        in_size = hparams.get("in_size", 100)
        hidden_size = hparams.get("hidden_size", 50)
        activation = hparams.get("activation", True)

        # Build a minimal MLP with only the forward-pass layers, no metrics
        mlp_model = torch.nn.Module()
        mlp_model.mlp = LinearWithHidden(in_size, hidden_size, Element.n_elements, activation)
        mlp_model.forward = lambda x: mlp_model.mlp(x)

        # Load matching weights
        state_dict = {k: v for k, v in checkpoint["state_dict"].items()
                      if k.startswith("mlp.")}
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        mlp_vector_length = in_size
        logger.info(f"  MLP loaded (fallback, in_size={in_size})")
    except Exception as e:
        logger.warning(f"  Could not load MLP: {e}")


# --- Request / Response schemas ---


class DeisotopeRequest(BaseModel):
    masses: List[float]
    intensities: List[float]
    model: str = "xgb"
    threshold: float = 0.5


class DeisotopeResponse(BaseModel):
    labels: List[int]
    peak_masses: List[float]
    peak_intensities: List[float]


class CheckFormulaRequest(BaseModel):
    masses: List[float]
    intensities: List[float]
    formula: str
    charge: int = 1
    cal_error: Optional[float] = None
    dist_error: Optional[float] = None


class CheckFormulaResponse(BaseModel):
    cosine_distance: float
    matched_percentage: float
    mass_error_ppm: float
    matched_masses: List[float]
    matched_intensities: List[float]


class PredictFormulaRequest(BaseModel):
    masses: List[float]
    intensities: List[float]
    peak_indices: List[int]


class ElementPrediction(BaseModel):
    symbol: str
    count: float


class PredictFormulaResponse(BaseModel):
    elements: List[ElementPrediction]


# --- Endpoints ---


@app.get("/health")
def health():
    return {
        "status": "ok",
        "deisotoping_models": list(deisotoping_models.keys()),
        "mlp_loaded": mlp_model is not None,
    }


@app.post("/deisotope", response_model=DeisotopeResponse)
def deisotope(req: DeisotopeRequest):
    if req.model not in deisotoping_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {list(deisotoping_models.keys())}",
        )

    spectrum = Spectrum(
        masses=np.array(req.masses),
        ints=np.array(req.intensities),
    )

    deisotoper = deisotoping_models[req.model]
    labels = deisotoper.run(spectrum, threshold=req.threshold)

    return DeisotopeResponse(
        labels=[int(l) for l in labels],
        peak_masses=req.masses,
        peak_intensities=req.intensities,
    )


@app.post("/check-formula", response_model=CheckFormulaResponse)
def check_formula(req: CheckFormulaRequest):
    spectrum = Spectrum(
        masses=np.array(req.masses),
        ints=np.array(req.intensities),
    )
    formula = Formula(req.formula, charge=req.charge)

    kwargs = {}
    if req.cal_error is not None:
        kwargs["cal_error"] = req.cal_error
    if req.dist_error is not None:
        kwargs["dist_error"] = req.dist_error

    try:
        cos_dist, real_coords, matched_pct, mass_error = check_presence(
            spectrum, formula, **kwargs
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    def _safe_float(v):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f

    return CheckFormulaResponse(
        cosine_distance=_safe_float(cos_dist),
        matched_percentage=_safe_float(matched_pct),
        mass_error_ppm=_safe_float(mass_error),
        matched_masses=[_safe_float(m) for m in real_coords[0]],
        matched_intensities=[_safe_float(i) for i in real_coords[1]],
    )


@app.post("/predict-formula", response_model=PredictFormulaResponse)
def predict_formula(req: PredictFormulaRequest):
    if mlp_model is None:
        raise HTTPException(status_code=503, detail="MLP model not loaded")

    spectrum = Spectrum(
        masses=np.array(req.masses),
        ints=np.array(req.intensities),
    )

    rid = RealIsotopicDistribution(spectrum, req.peak_indices)
    # vectorize() produces n_bins-1 elements, so pass length+1 to get the
    # correct input dimension for the MLP.
    representations = rid.get_representation(length=mlp_vector_length + 1)

    # Stack all peak vectors into a single input tensor
    vectors = [rep[0] for rep in representations]
    input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32)

    with torch.no_grad():
        output = mlp_model(input_tensor)

    # Average predictions across peaks in the cluster
    avg_output = output.mean(dim=0).numpy()

    # Map to element symbols, filtering to significant predictions
    elements = []
    for idx in range(len(avg_output)):
        count = float(avg_output[idx])
        element_number = idx + 1  # ELEMENT_DICT keys are 1-indexed
        if count > 0.5 and element_number in ELEMENT_DICT:
            elements.append(
                ElementPrediction(symbol=ELEMENT_DICT[element_number], count=count)
            )

    return PredictFormulaResponse(elements=elements)
