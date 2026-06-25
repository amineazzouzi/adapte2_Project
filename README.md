# Adapte2 — Analyse de Signaux Électriques Industriels

Détection et classification d'anomalies électriques sur des exploitations agricoles à partir de signaux TRMS (1 s) et oscilloscope (200 ms / 50 kHz).

---

## Architecture du projet

```
adapte2_project/
├── data_to_data_lake/main.py   # ETL : .txt bruts → Data Lake Parquet
├── oscillo_analysis.py          # Pipeline 2 : groupement de fenêtres oscilloscope
└── trms_analysis.py             # Pipeline 1 : détection pics + ruptures TRMS
```

**Dossiers de données** : placer les exports `.txt` / `.csv` dans `data_/` (oscillo) ou dans la structure attendue par `DATA_DIR` (TRMS). Les sorties vont dans `outputs/` — **effacé et recréé à chaque exécution**.

---

## Prérequis

```bash
pip install pandas numpy scipy matplotlib seaborn ruptures scikit-learn pyarrow
# GPU (optionnel pour oscillo_analysis.py, obligatoire pour trms_analysis.py) :
pip install cupy cudf cuml  # RAPIDS — nécessite CUDA
```

---

## ETL : `data_to_data_lake/main.py`

Ingère tous les `.txt` bruts d'une arborescence vers un Data Lake Parquet partitionné.

```bash
python data_to_data_lake/main.py
```

- **Détection du type** : si `TRMS` / `trms` est dans le chemin → fichier TRMS, sinon oscilloscope.
- **Détection du boîtier** : regex `boitier_\d+` dans le chemin.
- **Structure de sortie** : `data_lake/boitier_x/{oscillo,TRMS}/voie_x/year=Y/month=M/day=D/data.parquet`
- Les fichiers Parquet existants sont fusionnés et dédupliqués sur `timestamp`.
- Lecture chirurgicale colonne par colonne (une voie à la fois) pour économiser la RAM.

Constantes à adapter : `SOURCE_ROOT_DIR`, `DATA_LAKE_DIR`.

---

## Pipeline 1 : `trms_analysis.py` — Détection Pics & Ruptures

> ⚠️ **Requiert un GPU CUDA** — imports directs `cudf`, `cupy`, `cuml`, aucun fallback CPU.

```bash
python trms_analysis.py
```

### Fonctionnement en 3 blocs hybrides

| Bloc | Exécution | Ce qu'il fait |
|------|-----------|---------------|
| **1 — Pics** | GPU (cuDF/CuPy) | Segmente le signal (`WINDOW_SIZE_PEAK` = 3600 pts) ; signale les fenêtres où `\|max−min\| > PEAK_MAX_MIN_DIFF_THRESHOLD` ou `efficace > PEAK_EFFICACE_THRESHOLD` |
| **2 — Ruptures** | CPU (pandas/ruptures) | Dérivée médiane glissante → candidats → KMeans(k=2) pour filtrer les faux positifs → Binary Segmentation locale pour affiner ; traitement par blocs de 250 000 lignes (protection RAM) |
| **3 — Oscilloscope HR** | GPU + CPU (cuDF + scipy) | Relie chaque timestamp de pic au fichier oscilloscope couvrant la plage (30 min) ; extrait ±1 s ; classe la fenêtre via 3 features |

### Sorties

```
outputs/
└── anomalie_voie_{v}/
    ├── anomalies_pics/       # .png par pic détecté
    └── anomalies_ruptures/   # .png global + zoom par rupture
```

### Classification des fenêtres (Bloc 3)

| Feature | Bruit normal | Anomalie sinusoïdale |
|---------|-------------|----------------------|
| Entropie spectrale | ~0.95 (énergie diffuse) | ~0.002 (énergie concentrée) |
| Ratio puissance dominante (RPD) | ~0.002 | ~0.99 |
| Score d'autocorrélation (1–50 ms) | ~0.03 | ~0.95 |

### Convention de nommage (Bloc 3)

Le couplage TRMS↔oscilloscope repose sur le nom des fichiers :
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

## Pipeline 2 : `oscillo_analysis.py` — Groupement par similarité

> GPU CuPy si disponible, fallback NumPy automatique sinon.

```bash
python oscillo_analysis.py
```

### Fonctionnement

1. **Chargement** (`load_all_oscillo_files`) : scan récursif des `.txt/.csv`, gestion des coupures temporelles > 0,5 s, padding/troncature à `TARGET_PTS` = 5000 points.
2. **NCC batch GPU** (`compute_max_ncc_batch_gpu`) : corrélation croisée de N paires consécutives en un seul appel FFT via `_xcorr_fft_batch`. Si NCC ≥ `NCC_THRESHOLD` entre deux fenêtres consécutives → même groupe d'événements.
3. **Fréquence dominante** (`compute_dominant_frequency_batch_gpu`) : somme de Fourier sur grille de fréquences = produit matriciel GPU ; `FREQ_CHUNK` borne la VRAM utilisée par bloc.
4. **Export** : plots `.png` parallélisés via `ProcessPoolExecutor`, `recapitulatif_anomalies.csv`, `rapport_anomalies.html` interactif.

### Sorties

```
outputs/
├── evenement_XX_debut_YYYY-MM-DD_.../
│   ├── window_XXXX_REFERENCE_…Hz.png
│   └── window_XXXX_ncc_0.XXXX_…Hz.png
├── recapitulatif_anomalies.csv
└── rapport_anomalies.html
```

### Constantes clés

| Constante | Rôle |
|-----------|------|
| `DATA_PATH` | Dossier d'entrée (scan récursif) |
| `NCC_THRESHOLD` | Seuil de similarité pour regrouper (défaut 0.90) |
| `NCC_MAX_LAG` | Décalage max pour la recherche NCC (samples) |
| `GPU_BATCH_SIZE` | Paires NCC par batch GPU (défaut 256 — réduire si OOM) |
| `FREQ_CHUNK` | Fréquences par bloc GPU (défaut 1000 — réduire si OOM) |
| `NUM_WORKERS` | Workers multiprocessing pour l'export plots (défaut `min(CPU, 8)`) |
