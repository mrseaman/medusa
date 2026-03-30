# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MEDUSA (Machine Learning Enabled Deisotoping and Untargeted Spectra Analysis) is a Python framework for high-resolution mass spectrometry data analysis. It combines ML (xgboost, catboost, random forest, LSTM, MLP) with traditional MS algorithms for deisotoping, molecular formula determination, and sample identification.

## Common Commands

```bash
# Install dependencies
make init              # or: pip install -r requirements.txt

# Run all tests
make test              # or: py.test tests

# Run a single test file
py.test tests/test_formula.py

# Run a single test
py.test tests/test_formula.py::TestIsotopicDistribution::test_formula_isodistribution

# Run the API server
uvicorn server:app --host 0.0.0.0 --port 8000

# Build Sphinx docs
cd docs && make html
```

## Architecture

### Core Package: `mass_automation/`

**`experiment.py`** — Central module. `Spectrum` (single mass spectrum with peak picking, vectorization) and `Experiment` (full MS run with chromatogram, summarization). Entry point for loading `.mzXML` and `.pkl` data.

**`formula/`** — Formula analysis pipeline:
- `__init__.py`: `Formula` class (isotopic distribution calc via pyteomics, arithmetic ops, 119-element vector representation) and `RealIsotopicDistribution` (extracts/vectorizes experimental isotope patterns)
- `model.py`: PyTorch Lightning `LSTM` and `MLP` models for element classification/regression
- `check_formula.py`: `check_presence()` verifies formula against spectrum via cosine distance
- `determination.py`: `brute_force_search()` enumerates formulas matching exact mass within tolerance

**`deisotoping/process.py`** — `Deisotoper` ABC with two implementations:
- `LinearDeisotoper`: rule-based peak grouping
- `MlDeisotoper`: ML-based (loads models from `data/models/`). Both are callable, returning cluster label arrays.

**`sample_identification/`** — PCA/t-SNE clustering and visualization for sample comparison.

**`utils.py`** — `Element` class (periodic table constants), `ELEMENT_DICT`, monoisotopic element list.

### Data Flow

```
MS Data (mzXML/pkl) → Spectrum → Peak Detection → Deisotoping → Isotopic Analysis
→ Vectorization (500D) → Neural Network → Element Predictions → Formula Verification
```

### API Server: `server.py`

FastAPI service with three endpoints: `/deisotope`, `/check-formula`, `/predict-formula`. Loads deisotoping models (xgb, rf, cb) and MLP formula model at startup. Pydantic request/response models defined inline.

### Pre-trained Models

- `data/models/`: xgboost, random forest, catboost deisotoping models
- `nn_models/regression/`: MLP checkpoints for formula prediction
- `nn_models/classification/`: LSTM checkpoints for element classification

### Test Data

Test spectra (pickle and mzXML) are in `data/testing_spectra/`. Tests load these directly — no fixtures or mocking of data.

## Key Dependencies

- `pyopenms` — MS file I/O (optional, stubbed in server.py)
- `pyteomics` — isotopic distribution calculations
- `chemparse` — molecular formula string parsing
- `catboost`, `xgboost` — ML deisotoping models
- `pytorch-lightning`, `torch` — neural network training/inference
