import logging
import math
import os
import sys
import types
import uuid
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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

# --- Server-side spectrum store ---

spectra_store: Dict[str, dict] = {}


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

        mlp_model = torch.nn.Module()
        mlp_model.mlp = LinearWithHidden(in_size, hidden_size, Element.n_elements, activation)
        mlp_model.forward = lambda x: mlp_model.mlp(x)

        state_dict = {k: v for k, v in checkpoint["state_dict"].items()
                      if k.startswith("mlp.")}
        mlp_model.load_state_dict(state_dict)
        mlp_model.eval()
        mlp_vector_length = in_size
        logger.info(f"  MLP loaded (fallback, in_size={in_size})")
    except Exception as e:
        logger.warning(f"  Could not load MLP: {e}")


# --- Helpers ---


def _resolve_spectrum(masses, intensities, spectrum_id) -> Spectrum:
    if spectrum_id:
        entry = spectra_store.get(spectrum_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Unknown spectrum_id: {spectrum_id}")
        return entry["spectrum"]
    if not masses or not intensities:
        raise HTTPException(status_code=400, detail="Provide either spectrum_id or both masses and intensities")
    if len(masses) != len(intensities):
        raise HTTPException(
            status_code=400,
            detail=f"masses ({len(masses)}) and intensities ({len(intensities)}) must have the same length",
        )
    return Spectrum(masses=np.array(masses), ints=np.array(intensities))


# --- Request / Response schemas ---


class DeisotopeRequest(BaseModel):
    spectrum_id: Optional[str] = None
    masses: Optional[List[float]] = None
    intensities: Optional[List[float]] = None
    model: str = "cb"
    threshold: float = 0.5


class ClusterInfo(BaseModel):
    cluster_id: int
    peak_indices: List[int]
    masses: List[float]
    intensities: List[float]
    monoisotopic_mass: float


class DeisotopeResponse(BaseModel):
    num_peaks: int
    num_clusters: int
    num_unassigned: int
    clusters: List[ClusterInfo]


class CheckFormulaRequest(BaseModel):
    spectrum_id: Optional[str] = None
    masses: Optional[List[float]] = None
    intensities: Optional[List[float]] = None
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


class ClusterInput(BaseModel):
    cluster_id: int
    peak_indices: List[int]


class PredictFormulaRequest(BaseModel):
    spectrum_id: Optional[str] = None
    masses: Optional[List[float]] = None
    intensities: Optional[List[float]] = None
    peak_indices: Optional[List[int]] = None
    clusters: Optional[List[ClusterInput]] = None


class ElementPrediction(BaseModel):
    symbol: str
    count: float


class FormulaPrediction(BaseModel):
    cluster_id: int
    monoisotopic_mass: float
    suggested_formula: str
    mass_error_ppm: float
    elements: List[ElementPrediction]


class PredictFormulaResponse(BaseModel):
    predictions: List[FormulaPrediction]


class LoadPeaklistRequest(BaseModel):
    file_path: str
    mass_column: Optional[str] = None
    intensity_column: Optional[str] = None
    sheet_name: Optional[str] = None


class LoadPeaklistResponse(BaseModel):
    spectrum_id: str
    num_peaks: int
    mass_range: List[float]
    max_intensity: float
    columns: List[str]


# --- Endpoints ---


@app.get("/health")
def health():
    return {
        "status": "ok",
        "deisotoping_models": list(deisotoping_models.keys()),
        "mlp_loaded": mlp_model is not None,
        "loaded_spectra": len(spectra_store),
    }


@app.post("/deisotope", response_model=DeisotopeResponse)
def deisotope(req: DeisotopeRequest):
    spectrum = _resolve_spectrum(req.masses, req.intensities, req.spectrum_id)

    if req.model not in deisotoping_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model}'. Available: {list(deisotoping_models.keys())}",
        )

    deisotoper = deisotoping_models[req.model]
    labels = deisotoper.run(spectrum, threshold=req.threshold)

    cluster_peaks: dict = {}
    num_unassigned = 0
    for idx, label in enumerate(labels):
        label = int(label)
        if label < 0:
            num_unassigned += 1
            continue
        cluster_peaks.setdefault(label, []).append(idx)

    clusters = []
    for cid, indices in sorted(cluster_peaks.items()):
        c_masses = [float(spectrum.masses[i]) for i in indices]
        c_ints = [float(spectrum.ints[i]) for i in indices]
        clusters.append(ClusterInfo(
            cluster_id=cid,
            peak_indices=indices,
            masses=c_masses,
            intensities=c_ints,
            monoisotopic_mass=c_masses[0],
        ))

    clusters.sort(key=lambda c: c.intensities[0], reverse=True)

    return DeisotopeResponse(
        num_peaks=len(labels),
        num_clusters=len(clusters),
        num_unassigned=num_unassigned,
        clusters=clusters,
    )


@app.post("/check-formula", response_model=CheckFormulaResponse)
def check_formula(req: CheckFormulaRequest):
    spectrum = _resolve_spectrum(req.masses, req.intensities, req.spectrum_id)
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


def _build_formula_string(elements: List[ElementPrediction]) -> str:
    parts: dict = {}
    for e in elements:
        n = round(e.count)
        if n > 0:
            parts[e.symbol] = n
    ordered = []
    if "C" in parts:
        ordered.append(("C", parts.pop("C")))
    if "H" in parts:
        ordered.append(("H", parts.pop("H")))
    for sym in sorted(parts.keys()):
        ordered.append((sym, parts[sym]))
    return "".join(f"{sym}{n}" if n > 1 else sym for sym, n in ordered)


def _predict_one_cluster(spectrum, peak_indices, cluster_id):
    rid = RealIsotopicDistribution(spectrum, peak_indices)
    representations = rid.get_representation(length=mlp_vector_length + 1)
    vectors = [rep[0] for rep in representations]
    input_tensor = torch.tensor(np.array(vectors), dtype=torch.float32)

    with torch.no_grad():
        output = mlp_model(input_tensor)

    avg_output = output.mean(dim=0).numpy()

    elements = []
    for idx in range(len(avg_output)):
        count = float(avg_output[idx])
        element_number = idx + 1
        if count > 0.5 and element_number in ELEMENT_DICT:
            elements.append(ElementPrediction(symbol=ELEMENT_DICT[element_number], count=count))

    formula_str = _build_formula_string(elements)
    mono_mass = float(spectrum.masses[peak_indices[0]])

    mass_error = 0.0
    if formula_str:
        try:
            theoretical = Formula(formula_str).monoisotopic_mass
            mass_error = abs(theoretical - mono_mass) / mono_mass * 1e6
        except Exception:
            pass

    return FormulaPrediction(
        cluster_id=cluster_id,
        monoisotopic_mass=mono_mass,
        suggested_formula=formula_str,
        mass_error_ppm=round(mass_error, 3),
        elements=elements,
    )


@app.post("/predict-formula", response_model=PredictFormulaResponse)
def predict_formula(req: PredictFormulaRequest):
    spectrum = _resolve_spectrum(req.masses, req.intensities, req.spectrum_id)

    if mlp_model is None:
        raise HTTPException(status_code=503, detail="MLP model not loaded")

    if not req.clusters and not req.peak_indices:
        raise HTTPException(status_code=400, detail="Provide clusters or peak_indices")

    cluster_inputs = req.clusters or [ClusterInput(cluster_id=0, peak_indices=req.peak_indices)]
    predictions = [_predict_one_cluster(spectrum, c.peak_indices, c.cluster_id) for c in cluster_inputs]

    return PredictFormulaResponse(predictions=predictions)


MASS_COLUMN_NAMES = ["m/z", "mz", "mass", "t_mass", "measured mass", "obs. m/z", "m_z"]
INTENSITY_COLUMN_NAMES = ["intensity", "int", "i", "abs. intensity", "rel. intensity", "height"]


def _guess_column(columns, candidates):
    lower = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


@app.post("/load-peaklist", response_model=LoadPeaklistResponse)
def load_peaklist(req: LoadPeaklistRequest):
    fp = req.file_path
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail=f"File not found: {fp}")

    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(fp, sheet_name=req.sheet_name or 0)
        elif ext == ".csv":
            df = pd.read_csv(fp)
        elif ext in (".tsv", ".txt"):
            df = pd.read_csv(fp, sep="\t")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}. Use .xlsx, .csv, or .tsv")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to read file: {e}")

    mass_col = req.mass_column or _guess_column(df.columns, MASS_COLUMN_NAMES)
    int_col = req.intensity_column or _guess_column(df.columns, INTENSITY_COLUMN_NAMES)

    if mass_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify mass column. Columns: {list(df.columns)}. Set mass_column explicitly.",
        )
    if int_col is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not identify intensity column. Columns: {list(df.columns)}. Set intensity_column explicitly.",
        )

    df = df.dropna(subset=[mass_col, int_col])
    masses = df[mass_col].astype(float).values
    intensities = df[int_col].astype(float).values

    if len(masses) == 0:
        raise HTTPException(status_code=422, detail="No valid peaks found in file")

    sid = uuid.uuid4().hex[:12]
    spectra_store[sid] = {
        "spectrum": Spectrum(masses=masses, ints=intensities),
        "file": fp,
    }

    return LoadPeaklistResponse(
        spectrum_id=sid,
        num_peaks=len(masses),
        mass_range=[float(masses.min()), float(masses.max())],
        max_intensity=float(intensities.max()),
        columns=list(df.columns),
    )
