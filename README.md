# Adapte2 — Analyse de Signaux Électriques Industriels

Détection et classification d'anomalies électriques sur des exploitations agricoles à partir de signaux TRMS (1 s) et oscilloscope (200 ms / 50 kHz).

---

## Architecture du projet

Toute la logique vit dans `src/`. Il y a **un seul point d'entrée utilisateur : `interface.py`**. `to_data_lake.py` reste un script séparé car c'est une étape distincte (ingestion des données brutes), pas de l'analyse.

```
adapte2_project/
├── interface.py              # ★ SEUL POINT D'ENTRÉE — GUI Tkinter
├── to_data_lake.py           # ETL séparé : .txt bruts → Data Lake Parquet
│
├── scripts/                  # Workers CLI internes — jamais lancés directement,
│   ├── oscillo_analysis.py   #   invoqués en subprocess isolé (un par job/GPU)
│   └── oscillo_correlation.py#   par src/analysis/batch_pipeline.py et src/ui/app.py
│
├── src/
│   ├── core/                 # config.py (dataclasses de config), gpu.py (détection CuPy), paths.py
│   ├── io/                   # datalake_reader/writer.py, raw_file_loader.py, profile_io.py
│   ├── signal_processing/    # filtering.py, frequency.py, ncc.py (GPU), ncc_cpu.py, windowing.py
│   ├── analysis/             # oscillo_pipeline.py, correlation_pipeline.py, batch_pipeline.py,
│   │                         # event_tracking.py, clustering.py, type_registry.py
│   ├── reporting/            # html_report.py, plots.py, palette.py
│   └── ui/                   # app.py (implémentation Tkinter), signal_row.py, timeline_widget.py
│
└── ancienne_version/         # (gitignored) archive — trms_analysis.py (Pipeline 1) n'est PAS
                               # encore migré vers src/ : script monolithique historique, imports
                               # GPU en dur, chemins à adapter, à lancer séparément.
```

**Données** : `data_lake/boitier_x/{oscillo,TRMS}/voie_x/year=Y/month=M/day=D/data.parquet`, alimenté par `to_data_lake.py`. Les sorties oscillo vont dans `results/outputs_{boitier}_v{voie}/` (un dossier par signal, remplacé à chaque nouveau run — pas un dossier par jour) ; la corrélation dans `results/outputs_correlation/`. `outputs/` reste réservé à `trms_analysis.py` (legacy), **effacé et recréé à chaque exécution**.

---

## Prérequis

```bash
pip install -r requirements.txt
# GPU (optionnel pour l'analyse oscillo/ETL, obligatoire pour trms_analysis.py legacy) :
pip install cupy-cuda12x cudf-cu12 cuml-cu12  # RAPIDS — nécessite CUDA
```

`src/core/gpu.py` teste un device CUDA réel (pas juste la présence du package) et expose `GPU_AVAILABLE` ; c'est la source unique de fallback GPU→CPU pour l'analyse oscillo et l'ETL.

---

## Utilisation

### 1. Ingestion des données — `to_data_lake.py`

Avant toute analyse, ingérer les `.txt` bruts vers le Data Lake Parquet partitionné (logique dans `src/io/datalake_writer.py`) :

```bash
python to_data_lake.py
python to_data_lake.py --source-root-dir /chemin/brut --data-lake-dir /chemin/data_lake
```

- **Détection du type** : `TRMS`/`trms` dans le chemin → fichier TRMS, sinon oscilloscope.
- **Détection du boîtier** : regex `boitier_\d+` dans le chemin.
- Fichiers Parquet existants fusionnés et dédupliqués sur `timestamp` ; lecture chirurgicale (une voie à la fois) pour économiser la RAM.

### 2. Analyse — `interface.py` (point d'entrée unique)

```bash
python interface.py
```

Workflow (implémentation complète dans `src/ui/app.py`) :

1. Configurer N signaux (boîtier, voie, date début → date fin) dans l'interface.
2. **Lancer** → tous les signaux sont analysés **en parallèle**, un process Python isolé par job, un GPU par process en round-robin (`src/analysis/batch_pipeline.OscilloBatchRunner`, `CUDA_VISIBLE_DEVICES`). Chaque job exécute `scripts/oscillo_analysis.py` : chargement des fenêtres depuis le data lake, détection d'anomalies par amplitude, groupement en événements (NCC temporelle), clustering en types (NCC inter-événements), rattachement à une base de types persistante inter-runs, export `rapport_enrichi.html` + `signal_profile.json`.
3. Si ≥ 2 signaux : `scripts/oscillo_correlation.py` s'exécute ensuite sur l'ensemble — détecte les types d'anomalies partagés entre signaux (NCC entre fenêtres de référence), exporte `rapport_correlation.html`.
4. Journal en temps réel, timeline Gantt des anomalies, boutons pour ouvrir les rapports.

L'isolation en process séparé (plutôt qu'un run direct en Python dans l'interface) garantit qu'un pipeline qui plante ne casse pas la fenêtre, et que la VRAM CUDA est proprement libérée entre deux runs — voir les docstrings de `src/analysis/batch_pipeline.py` et `src/ui/app.py`.

> `scripts/oscillo_analysis.py` et `scripts/oscillo_correlation.py` ne sont **pas** des points d'entrée : ce sont des workers internes invoqués via `python -m scripts.<nom>` avec `cwd` = racine du projet. Les lancer à la main pour du débogage reste possible (`python -m scripts.oscillo_analysis --boitier boitier_1 --voie 1 --dates 2026-03-17`), mais l'usage normal passe par `interface.py`.

---

## Pipeline 1 (legacy) : `ancienne_version/trms_analysis.py` — Détection Pics & Ruptures

> ⚠️ **Non migré vers `src/`, non branché sur `interface.py`** — script monolithique historique, imports GPU en dur (`cudf`, `cupy`, `cuml`, aucun fallback CPU), chemins codés en dur à adapter avant exécution.

```bash
cd ancienne_version && python trms_analysis.py
```

### Fonctionnement en 3 blocs hybrides

| Bloc | Exécution | Ce qu'il fait |
|------|-----------|---------------|
| **1 — Pics** | GPU (cuDF/CuPy) | Segmente le signal (`WINDOW_SIZE_PEAK` = 3600 pts) ; signale les fenêtres où `\|max−min\| > PEAK_MAX_MIN_DIFF_THRESHOLD` ou `efficace > PEAK_EFFICACE_THRESHOLD` |
| **2 — Ruptures** | CPU (pandas/ruptures) | Dérivée médiane glissante → candidats → KMeans(k=2) pour filtrer les faux positifs → Binary Segmentation locale pour affiner ; traitement par blocs de 250 000 lignes (protection RAM) |
| **3 — Oscilloscope HR** | GPU + CPU (cuDF + scipy) | Relie chaque timestamp de pic au fichier oscilloscope couvrant la plage (30 min) ; extrait ±1 s ; classe la fenêtre via 3 features |

### Classification des fenêtres (Bloc 3)

| Feature | Bruit normal | Anomalie sinusoïdale |
|---------|-------------|----------------------|
| Entropie spectrale | ~0.95 (énergie diffuse) | ~0.002 (énergie concentrée) |
| Ratio puissance dominante (RPD) | ~0.002 | ~0.99 |
| Score d'autocorrélation (1–50 ms) | ~0.03 | ~0.95 |

### Convention de nommage (Bloc 3)

- Fichiers oscilloscope : `YYYY-MM-DD_HH-MM.txt` (couverture 30 min depuis l'heure)
- PNGs de pics générés : `…timestamp_YYYY-MM-DD_HH-MM-SS.png`

Tout fichier ne respectant pas ces patterns est silencieusement ignoré.

### Constantes clés

| Constante | Rôle |
|-----------|------|
| `DATA_DIR` / `DATA_FILE` / `OSCILLO_DIR` | Chemins d'entrée |
| `NUM_VOIES` | Nombre de voies à traiter (1–3) |
| `PEAK_MAX_MIN_DIFF_THRESHOLD` | Seuil amplitude min−max (défaut 100) |
| `PEAK_EFFICACE_THRESHOLD` | Seuil RMS (défaut 20) |
| `WINDOW_SIZE_PEAK` | Taille segment de détection de pics (défaut 3600) |

---

## Configuration (Pipeline 2 & corrélation)

Tous les défauts vivent dans `src/core/config.py` (dataclasses `SignalConfig`, `DataLakeConfig`, `BatchConfig`, `CorrelationConfig`), en chemins **relatifs à la racine du projet** (tous les points d'entrée tournent avec `cwd` = racine).

| Constante | Dataclass | Rôle |
|-----------|-----------|------|
| `ncc_threshold` | `SignalConfig` | Seuil NCC pour regrouper deux fenêtres consécutives en événement |
| `ncc_max_lag` | `SignalConfig` | Décalage max pour la recherche NCC (samples) |
| `ncc_type_threshold` | `SignalConfig` | Seuil NCC pour le clustering inter-événements en types |
| `lowpass_cutoff_hz` | `SignalConfig` | Coupure (Hz) du filtre passe-bas appliqué avant tout calcul NCC |
| `global_type_ncc_threshold` | `SignalConfig` | Seuil NCC (signal brut) pour rattacher un type local à la base persistante |
| `gpu_batch_size` / `freq_chunk` | `SignalConfig` | Taille des batches GPU — réduire si OOM |
| `max_parallel` | `BatchConfig` | Jobs simultanés (0 = auto, un par GPU détecté) |
| `corr_window_s` | `CorrelationConfig` | Fenêtre de co-occurrence temporelle (±s) |
