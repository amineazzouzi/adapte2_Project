"""Configuration du pipeline oscillo — les défauts reproduisent exactement
les constantes actuellement codées en dur dans oscillo_analysis.py, pour
qu'un run sans arguments (CLI) reste identique à avant la réorganisation."""

from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    # ── Source ────────────────────────────────────────────────────────
    use_datalake: bool = True
    data_lake_path: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project/data_lake"
    boitier: str = "boitier_1"
    voie: int = 1
    dates: list = field(default_factory=lambda: ["2026-03-17"])

    # Mode fichiers bruts (si use_datalake=False)
    data_path: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project/data_lake"
    output_dir_raw: str = "outputs"
    channel_index: int = 0

    # ── Seuils NCC / détection ───────────────────────────────────────
    ncc_threshold: float = 0.30
    ncc_max_lag: int = 5000
    ncc_type_threshold: float = 0.3
    n_freq_bins: int = 100000

    # ── Filtrage passe-bas (avant calcul de similarité NCC) ──────────────
    lowpass_cutoff_hz: float = 3000.0

    # ── Rapport détaillé par type ─────────────────────────────────────
    type_hist_bin_min: int = 10

    # ── Perf ──────────────────────────────────────────────────────────
    gpu_batch_size: int = 2048
    freq_chunk: int = 100000
    num_workers: int = 4


@dataclass
class DataLakeConfig:
    """Défauts = constantes actuellement codées en dur dans to_data_lake.py."""
    source_root_dir: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project/Exploitation_Brehaudiere"
    data_lake_dir: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project/data_lake"
    chunk_size_base: int = 200_000
    gpu_sort_threshold: int = 300_000
    n_workers: int = 5


@dataclass
class CorrelationConfig:
    """Défauts = constantes actuellement codées en dur dans oscillo_correlation.py."""
    project_dir: str = "/home/azzouzi/Bureau/Projet IA/adapte2_Project"
    signals: list = field(default_factory=lambda: [
        {"boitier": "boitier_1", "voie": 1},
        {"boitier": "boitier_2", "voie": 1},
    ])
    output_dir_name: str = "results/outputs_correlation"
    corr_window_s: float = 30.0
    hist_bin_min: int = 1
    ncc_type_threshold: float = 0.70
