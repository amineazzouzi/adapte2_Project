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
    dates: list = field(default_factory=lambda: ["2026-02-18"])

    # Mode fichiers bruts (si use_datalake=False)
    data_path: str = "data_lake"
    output_dir_raw: str = "outputs"
    channel_index: int = 0

    # ── Filtrage anomalies par fenêtre (src/signal_processing/windowing.py) ──
    # 1. filter_anomaly_windows : amplitude locale moyenne (n_segments segments)
    anomaly_n_segments: int = 10
    anomaly_threshold: float = 50
    # 2. filter_by_peak_threshold : pic positif ET pic négatif (abs) au-dessus
    #    du seuil, combiné en ET avec le filtre ci-dessus
    peak_threshold: float = 30

    # ── Classification par nombre de pics isolés (windowing.py::classify_windows_by_peak_kmeans) ──
    # Méthode indépendante du filtrage ci-dessus : segmente la fenêtre, score
    # chaque segment par max(signal**4), sépare "pic" / "normal" par KMeans
    # (2 clusters) pour classer la fenêtre pic_1 / pic_2 / pic_3 / ... —
    # calculée avant le passe-bas et la NCC. Voir pic_kmeans_explorer.ipynb.
    peak_kmeans_search_width: int = 1           # taille des segments (échantillons)
    peak_kmeans_min_cluster_separation: float = 400  # centre_pic/centre_normal sous ce ratio -> pas de vrai pic

    # ── Distinction "type pic" / "type pic + forme" (windowing.py::compute_window_energy) ──
    # Parmi les fenêtres déjà classées "pic" ci-dessus, celles dont l'énergie
    # (somme(|signal filtré médian|)) dépasse peak_energy_threshold comptent
    # EN PLUS comme "type pic + forme" (flag additionnel, pas une reclassification
    # exclusive) — seuil validé interactivement dans pic_kmeans_explorer.ipynb.
    peak_energy_median_filter_size: int = 5
    peak_energy_threshold: float = 10000

    # ── Seuils NCC / détection ───────────────────────────────────────
    ncc_threshold: float = 0.8
    ncc_max_lag: int = 5000
    ncc_type_threshold: float = 0.8

    # ── Filtrage passe-bas (avant calcul de similarité NCC) ──────────────

    lowpass_cutoff_hz: float = 20000.0

    # ── Rapport détaillé par type ─────────────────────────────────────
    type_hist_bin_min: int = 10

    # ── Base de types persistante (globale, entre tous les runs) ─────────
    type_db_dir: str = "results/type_database"
    global_type_ncc_threshold: float = 0.8

    # ── Perf ──────────────────────────────────────────────────────────
    gpu_batch_size: int = 2048
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
    n_workers: int = 2


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
    ncc_type_threshold: float = 0.8
