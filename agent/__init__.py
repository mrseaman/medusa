"""MEDUSA Analysis Agent package.

Stubs pyopenms early so mass_automation imports cleanly even when the
native library is unavailable.
"""

import sys
import types

if "pyopenms" not in sys.modules:
    try:
        import pyopenms  # noqa: F401
    except (ImportError, OSError):
        _stub = types.ModuleType("pyopenms")
        _stub.MSExperiment = None
        _stub.MzXMLFile = None
        sys.modules["pyopenms"] = _stub
