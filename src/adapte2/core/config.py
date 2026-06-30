"""Dataclasses de configuration du pipeline Adapte2."""

from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    target_pts: int = 5000
    ncc_threshold: float = 0.30
    ncc_max_lag: int = 1000
    ncc_type_threshold: float = 0.30
    amplitude_threshold: float = 50
    n_segments: int = 10
    hist_bin_min: int = 30
    max_events_ncc_matrix: int = 150
    export_plots: bool = False  # coûteux en RAM, désactivé par défaut


@dataclass
class GpuConfig:
    batch_size: int = 2048
    freq_chunk: int = 50000
    n_freq_bins: int = 50000
    sort_threshold: int = 300_000


@dataclass
class PathConfig:
    data_lake: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project/data_lake"
    type_library: str = "type_library"
    outputs: str = "outputs"


@dataclass
class PipelineConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    num_workers: int = 4
    cooccurrence_window_s: float = 30.0
