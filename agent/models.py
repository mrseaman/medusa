"""Model registry — loads and holds all MEDUSA models."""

import logging
import sys
import types

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Stub pyopenms if not installed so mass_automation can import cleanly.
if "pyopenms" not in sys.modules:
    try:
        import pyopenms  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("pyopenms")
        _stub.MSExperiment = None
        _stub.MzXMLFile = None
        sys.modules["pyopenms"] = _stub

from mass_automation.deisotoping.process import MlDeisotoper
from mass_automation.formula.model import MLP, LinearWithHidden
from mass_automation.utils import Element


class ModelRegistry:
    def __init__(self):
        self.deisotoping_models: dict = {}
        self.mlp_model = None
        self.mlp_vector_length: int = 500

    def load_all(self):
        self._load_deisotoping()
        self._load_mlp()

    def _load_deisotoping(self):
        for name, path in [
            ("xgb", "data/models/xgb_model.pkl"),
            ("rf", "data/models/rf_model.pkl"),
            ("cb", "data/models/cb_model.pkl"),
        ]:
            try:
                self.deisotoping_models[name] = MlDeisotoper().load(path)
                logger.info(f"Loaded deisotoping model: {name}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    def _load_mlp(self):
        ckpt_path = "nn_models/regression/mlp_0.25.ckpt"
        try:
            self.mlp_model = MLP.load_from_checkpoint(ckpt_path, map_location="cpu")
            self.mlp_model.eval()
            logger.info("MLP loaded via load_from_checkpoint")
        except Exception:
            logger.info("Retrying MLP load (inference-only fallback)...")
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                hparams = checkpoint.get("hyper_parameters", {})
                in_size = hparams.get("in_size", 100)
                hidden_size = hparams.get("hidden_size", 50)
                activation = hparams.get("activation", True)

                model = torch.nn.Module()
                model.mlp = LinearWithHidden(in_size, hidden_size, Element.n_elements, activation)
                model.forward = lambda x: model.mlp(x)

                state_dict = {k: v for k, v in checkpoint["state_dict"].items()
                              if k.startswith("mlp.")}
                model.load_state_dict(state_dict)
                model.eval()
                self.mlp_model = model
                self.mlp_vector_length = in_size
                logger.info(f"MLP loaded (fallback, in_size={in_size})")
            except Exception as e:
                logger.warning(f"Could not load MLP: {e}")

    def get_deisotoper(self, name: str) -> MlDeisotoper:
        if name not in self.deisotoping_models:
            available = list(self.deisotoping_models.keys())
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        return self.deisotoping_models[name]
