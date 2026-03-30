"""Session state that persists across tool calls in a conversation."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

from mass_automation.experiment import Spectrum, Experiment


@dataclass
class SessionState:
    spectrum: Optional[Spectrum] = None
    experiment: Optional[Experiment] = None
    labels: Optional[np.ndarray] = None
    analysis_history: List[dict] = field(default_factory=list)

    # Metadata about loaded data
    source_path: Optional[str] = None

    def get_cluster_indices(self, cluster_id: int) -> List[int]:
        if self.labels is None:
            raise RuntimeError("No deisotoping results. Run deisotope first.")
        indices = np.where(self.labels == cluster_id)[0].tolist()
        if not indices:
            raise ValueError(f"Cluster {cluster_id} not found.")
        return indices

    def get_cluster_count(self) -> int:
        if self.labels is None:
            return 0
        unique = np.unique(self.labels)
        return int((unique >= 0).sum())

    def get_cluster_summary(self) -> List[Tuple[int, float, int]]:
        """Returns list of (cluster_id, monoisotopic_mass, num_peaks)."""
        if self.labels is None or self.spectrum is None:
            return []
        unique = np.unique(self.labels)
        summary = []
        for cid in sorted(unique):
            if cid < 0:
                continue
            cid = int(cid)
            idx = np.where(self.labels == cid)[0]
            mono_mass = float(self.spectrum.masses[idx[0]])
            summary.append((cid, mono_mass, len(idx)))
        return summary

    def reset(self):
        self.spectrum = None
        self.experiment = None
        self.labels = None
        self.analysis_history.clear()
        self.source_path = None
