"""Configuration du pipeline oscillo — les défauts reproduisent exactement
les constantes actuellement codées en dur dans oscillo_analysis.py, pour
qu'un run sans arguments (CLI) reste identique à avant la réorganisation."""

from dataclasses import dataclass, field


@dataclass
class SignalConfig:
    # ── Source ────────────────────────────────────────────────────────
    # Chemins relatifs à la racine du projet : tous les points d'entrée
    # (interface.py, to_data_lake.py, scripts/*.py) tournent avec cwd =
    # racine du projet, donc pas besoin de chemin absolu machine-spécifique.
    use_datalake: bool = True
    data_lake_path: str = "data_lake"
    boitier: str = "boitier_1"
    voie: int = 1
    dates: list = field(default_factory=lambda: ["2026-03-17"])

    # Mode fichiers bruts (si use_datalake=False)
    data_path: str = "data_lake"
    output_dir_raw: str = "outputs"
    channel_index: int = 0

    # ── Filtrage anomalies par fenêtre (src/signal_processing/windowing.py) ──
    # 1. filter_anomaly_windows : amplitude locale moyenne (n_segments segments)
    anomaly_n_segments: int = 10
    anomaly_threshold: float = 0.0
    # 2. filter_by_peak_threshold : pic positif ET pic négatif (abs) au-dessus
    #    du seuil, combiné en ET avec le filtre ci-dessus
    peak_threshold: float = 25

    # ── Classification par nombre de pics isolés (windowing.py::classify_windows_by_peak_count) ──
    # Méthode indépendante du filtrage ci-dessus : compte les pics isolés
    # d'une fenêtre (point >> médiane locale du voisinage) pour la classer
    # pic_1 / pic_2 / pic_3 / ... — calculée avant le passe-bas et la NCC.
    peak_isolation_neighborhood: int = 20        # nb de points de part et d'autre pour la médiane locale
    peak_isolation_factor: float = 50            # point = pic si |point| > facteur * médiane(|voisinage|)
    peak_isolation_merge_gap: int = 50           # points-pics séparés de moins de X échantillons -> même pic
    peak_isolation_low_level_pct: float = 0.80   # fraction du signal qui doit rester sous le seuil bas
    peak_isolation_low_level_threshold: float = 25  # seuil "signal bas" (valeur absolue)

    # ── Seuils NCC / détection ───────────────────────────────────────
    ncc_threshold: float = 0.30
    ncc_max_lag: int = 5000
    ncc_type_threshold: float = 0.3
    n_freq_bins: int = 100000

    # ── Filtrage passe-bas (avant calcul de similarité NCC) ──────────────
    lowpass_cutoff_hz: float = 2500

    # ── Rapport détaillé par type ─────────────────────────────────────
    type_hist_bin_min: int = 10

    # ── Base de types persistante (globale, entre tous les runs) ─────────
    type_db_dir: str = "results/type_database"
    global_type_ncc_threshold: float = 0.70

    # ── Perf ──────────────────────────────────────────────────────────
    gpu_batch_size: int = 2048
    freq_chunk: int = 100000
    num_workers: int = 4


@dataclass
class DataLakeConfig:
    """source_root_dir pointe vers l'arborescence de données brutes (.txt) —
    spécifique à chaque environnement/serveur, à passer via --source-root-dir
    si elle ne vit pas à la racine du projet sous ce nom."""
    source_root_dir: str = "Exploitation_Brehaudiere"
    data_lake_dir: str = "data_lake"
    chunk_size_base: int = 200_000
    gpu_sort_threshold: int = 300_000
    n_workers: int = 5


@dataclass
class BatchConfig:
    """
    Liste explicite des jobs oscillo (boitier/voie/dates) à lancer en
    parallèle sur le serveur multi-GPU — un process par job, un GPU par
    process (round-robin sur le nombre de GPUs détecté). Les autres
    paramètres (seuils NCC, etc.) restent ceux par défaut de SignalConfig
    pour chaque job ; édite cette liste à la main pour un nouveau lot.
    """
    data_lake_path: str = "data_lake"
    jobs: list = field(default_factory=lambda: [
        {"boitier": "boitier_1", "voie": 1, "dates": ["2026-03-17"]},
        {"boitier": "boitier_1", "voie": 2, "dates": ["2026-03-17"]},
        {"boitier": "boitier_2", "voie": 1, "dates": ["2026-03-17"]},
    ])
    max_parallel: int = 1  # 1 = un seul job à la fois (GPU unique, comme avant la
                            # parallélisation) ; 0 = auto-détecté (nombre de GPUs)
    log_dir: str = "results/batch_logs"
    python_executable: str = ""  # "" = sys.executable (voir interface.py, qui préfère .venv/bin/python)


@dataclass
class CorrelationConfig:
    """project_dir par défaut = répertoire courant : tous les points d'entrée
    tournent avec cwd = racine du projet (interface.py le fixe explicitement
    via --project-dir pour scripts/oscillo_correlation.py)."""
    project_dir: str = "."
    signals: list = field(default_factory=lambda: [
        {"boitier": "boitier_1", "voie": 1},
        {"boitier": "boitier_2", "voie": 1},
    ])
    output_dir_name: str = "results/outputs_correlation"
    corr_window_s: float = 30.0
    hist_bin_min: int = 1
    ncc_type_threshold: float = 0.70
