#!/usr/bin/env python
# coding: utf-8
"""
╔═══════════════════════════════════════════════════════════════════════╗
║   PIPELINE GPU-ACCÉLÉRÉ — Groupement de Fenêtres d'Anomalies          ║
║   Version optimisée CuPy/GPU du notebook all_anomaly_pipeline.ipynb   ║
╚═══════════════════════════════════════════════════════════════════════╝

GOULOTS D'ÉTRANGLEMENT IDENTIFIÉS DANS LE NOTEBOOK ORIGINAL :
  1. compute_dominant_frequency : boucle Python pure sur 5000 fréquences,
     chacune sommant sur tout le signal (O(5000²) en Python pur) -> LENT
  2. compute_max_ncc : scipy.signal.correlate appelé séquentiellement
     pour CHAQUE paire de fenêtres consécutives -> répété N fois
  3. Boucle d'export : recalcule la NCC ET la freq dominante pour
     chaque fenêtre de chaque événement -> travail dupliqué

STRATÉGIE GPU :
  - compute_dominant_frequency_gpu : la somme de Fourier sur la grille de
    fréquences devient un produit matriciel (vectorisé) -> GPU massif
  - compute_max_ncc_gpu : corrélation croisée vectorisée via FFT sur GPU
  - Pré-calcul GPU en BATCH de toutes les fréquences dominantes et de
    toutes les NCC consécutives EN UNE SEULE FOIS (au lieu de recalculer
    pendant l'export) -> élimine la duplication de travail
  - Export des plots parallélisé en multiprocessing (CPU, car
    matplotlib n'est pas GPU-compatible)
"""

import os
import shutil
import glob
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # backend fichier sans display X11 (requis WSL/headless + multiprocessing)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import json
import cupy as cp # type: ignore
# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                      DÉTECTION GPU (CuPy)                             ║
# ╚═══════════════════════════════════════════════════════════════════════╝

try:
    # Test réel (pas juste l'import) pour s'assurer qu'un device répond
    cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    print(f"✅ GPU détectée : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception as e:
    cp = None
    GPU_AVAILABLE = False
    print(f"⚠️  GPU non disponible ({e}) — fallback CPU (NumPy)")

xp = cp if GPU_AVAILABLE else np

plt.rcParams['figure.figsize'] = (12, 4)
sns.set_theme(style="whitegrid")

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

# ── Mode source ─────────────────────────────────────────────────────
# True  → lit depuis le data lake Parquet (recommandé)
# False → lit depuis les fichiers txt/csv bruts (DATA_PATH)
USE_DATALAKE = True

# ── Config data lake (utilisé si USE_DATALAKE = True) ───────────────
DATA_LAKE_PATH   = "/home/aazzouzi/Projet IA/Adapte2/adapte2_project/data_lake"
DATALAKE_BOITIER = "boitier_1"   # "boitier_1" ou "boitier_2"
DATALAKE_VOIE    = 1             # 1, 2 ou 3
DATALAKE_DATE    = "2026-02-17"  # format YYYY-MM-DD

# ── Config fichiers bruts (utilisé si USE_DATALAKE = False) ─────────
DATA_PATH = "/home/aazzouzi/Projet IA/Adapte2/Groupement_Signaux_anomalie/data"

# OUTPUT_DIR est auto-calculé en mode data lake ; sinon utiliser la valeur ci-dessous
OUTPUT_DIR    = "outputs"
CHANNEL_INDEX = 0
TARGET_PTS    = 5000

NCC_THRESHOLD = 0.40
NCC_MAX_LAG = 500
N_FREQ_BINS = 5000          # Nombre de fréquences testées (comme l'original)

GPU_BATCH_SIZE = 256        # Nombre de paires NCC traitées par batch GPU (corrélation FFT)
FREQ_CHUNK = 1000           # Nombre de fréquences traitées simultanément par fenêtre
                             # (borne la mémoire : freq_chunk * L * 16 bytes par bloc)
                             # Exemple : 1000 * 5000 * 16 = 80 MB -> très sûr
                             # Si vous avez beaucoup de VRAM (>=24GB), montez à 5000
                             # (= tout d'un coup, plus rapide mais plus de mémoire)
NUM_WORKERS = min(os.cpu_count() or 4, 8)   # Pour export plots (CPU)

NCC_TYPE_THRESHOLD   = 0.4   # Seuil NCC pour fusionner deux événements en même type
HIST_BIN_MIN         = 30    # Largeur des bins temporels pour les histogrammes (minutes)
MAX_EVENTS_NCC_MATRIX = 150  # Au-delà, la matrice NCC n'est pas tracée (illisible)

# ╔═══════════════════════════════════════════════════════════════════════╗
# ║         [Cell 2] Chargement des fenêtres                              ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def load_from_datalake(datalake_path, boitier, voie, date, target_pts=TARGET_PTS):
    """
    Charge les données oscillo depuis le data lake Parquet et les transforme
    en fenêtres de target_pts points (même logique de gestion des gaps que
    load_all_oscillo_files : coupure si gap > 0.5s, pad/truncate à target_pts).
    """
    year, month, day = date.split('-')
    parquet_path = os.path.join(
        datalake_path, boitier, 'oscillo', f'voie_{voie}',
        f'year={year}', f'month={int(month):02d}', f'day={int(day):02d}',
        'data.parquet'
    )

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    print(f"--- Chargement data lake : {parquet_path} ---")
    df = pd.read_parquet(parquet_path)
    # df = df[:100*10000]
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Détecter les coupures (gaps > 0.5s) → autant de segments indépendants
    diffs = df['timestamp'].diff().dt.total_seconds().fillna(0)
    jump_indices  = np.where(diffs > 0.5)[0]
    start_indices = np.concatenate(([0], jump_indices))
    end_indices   = np.concatenate((jump_indices, [len(df)]))

    all_windows     = []
    all_time_arrays = []

    for s, e in zip(start_indices, end_indices):
        seg_sig = df['signal'].iloc[s:e].values.astype(np.float64)
        seg_ts  = df['timestamp'].iloc[s:e].values  # datetime64[ns]

        if len(seg_sig) < 100:
            continue

        if len(seg_sig) < target_pts:
            pad = target_pts - len(seg_sig)
            seg_sig = np.pad(seg_sig, (0, pad), 'constant')
            if len(seg_ts) > 1:
                dt = seg_ts[1] - seg_ts[0]
                seg_ts = np.concatenate([seg_ts,
                                          [seg_ts[-1] + (k + 1) * dt for k in range(pad)]])
            else:
                seg_ts = np.concatenate([seg_ts, [seg_ts[-1]] * pad])
        else:
            seg_sig = seg_sig[:target_pts]
            seg_ts  = seg_ts[:target_pts]

        all_windows.append(seg_sig)
        all_time_arrays.append(seg_ts)

    print(f"--- {len(all_windows)} fenêtres chargées "
          f"({boitier} / voie_{voie} / {date}) ---")
    return np.array(all_windows), np.array(all_time_arrays, dtype=object)

def _load_one_file(args):
    """Worker : charge un fichier oscillo et retourne ses segments (windows, time_arrays)."""
    file, channel_index, target_pts = args
    windows, time_arrs = [], []
    try:
        df = pd.read_csv(file, sep=None, engine='python', header=None)
        df[0] = pd.to_datetime(df[0])
        diffs         = df[0].diff().dt.total_seconds().fillna(0)
        jump_indices  = np.where(diffs > 0.5)[0]
        start_indices = np.concatenate(([0], jump_indices))
        end_indices   = np.concatenate((jump_indices, [len(df)]))
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) <= channel_index:
            return [], []
        val_col = num_cols[channel_index]
        for s, e in zip(start_indices, end_indices):
            window   = df[val_col].iloc[s:e].values
            time_arr = df[0].iloc[s:e].values
            if len(window) < 100:
                continue
            if len(window) < target_pts:
                pad = target_pts - len(window)
                window = np.pad(window, (0, pad), 'constant')
                if len(time_arr) > 1:
                    dt = time_arr[1] - time_arr[0]
                    time_arr = np.concatenate([time_arr,
                        [time_arr[-1] + (k + 1) * dt for k in range(pad)]])
                else:
                    time_arr = np.concatenate([time_arr, [time_arr[-1]] * pad])
            else:
                window   = window[:target_pts]
                time_arr = time_arr[:target_pts]
            windows.append(window)
            time_arrs.append(time_arr)
    except Exception:
        pass
    return windows, time_arrs


def load_all_oscillo_files(directory="data_", channel_index=0, target_pts=5000):
    print(f"--- Analyse du dossier : '{os.path.abspath(directory)}' ---")
    files = sorted([
        os.path.join(root, f)
        for root, _, names in os.walk(directory)
        for f in names
        if f.endswith(".txt") or f.endswith(".csv")
    ])

    all_windows, all_time_arrays = [], []
    args = [(f, channel_index, target_pts) for f in files]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for windows, time_arrs in executor.map(_load_one_file, args):
            all_windows.extend(windows)
            all_time_arrays.extend(time_arrs)

    print(f"\n--- Chargement terminé : {len(all_windows)} fenêtres récupérées ---")
    return np.array(all_windows), np.array(all_time_arrays)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║   [GPU] NCC vectorisée — toutes les paires consécutives en un batch   ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _xcorr_fft_batch(x_batch, y_batch, xp_mod):
    """
    Corrélation croisée 'same' pour un batch de paires (x_i, y_i) via FFT.
    x_batch, y_batch : (N, L) sur GPU ou CPU.
    Retourne corr : (N, L) équivalent à scipy.signal.correlate(x, y, mode='same')
    pour chaque paire, mais vectorisé sur tout le batch en un seul appel FFT.
    """
    N, L = x_batch.shape
    n_fft = 2 * L - 1
    # padding circulaire vers la taille FFT
    X = xp_mod.fft.rfft(x_batch, n=n_fft, axis=1)
    Y = xp_mod.fft.rfft(y_batch[:, ::-1], n=n_fft, axis=1)  # flip = corrélation
    corr_full = xp_mod.fft.irfft(X * Y, n=n_fft, axis=1)
    # extraire la zone centrale 'same' (taille L), comme scipy mode='same'
    start = (n_fft - L) // 2
    return corr_full[:, start:start + L]


def compute_max_ncc_batch_gpu(x_batch, y_batch, max_lag=500, xp_mod=None):
    """
    Version BATCH de compute_max_ncc : calcule la NCC max pour N paires
    de signaux simultanément sur GPU.
    x_batch, y_batch : arrays NumPy (N, L)
    Retourne : array NumPy (N,) des NCC max (0.0 si signal constant)
    """
    xp_mod = xp_mod or xp
    N, L = x_batch.shape

    xb = xp_mod.asarray(x_batch, dtype=xp_mod.float32)
    yb = xp_mod.asarray(y_batch, dtype=xp_mod.float32)

    x_c = xb - xb.mean(axis=1, keepdims=True)
    y_c = yb - yb.mean(axis=1, keepdims=True)
    std_x = xb.std(axis=1)
    std_y = yb.std(axis=1)

    corr = _xcorr_fft_batch(x_c, y_c, xp_mod)
    corr_norm = corr / (L * std_x[:, None] * std_y[:, None] + 1e-12)

    center = L // 2
    s = max(0, center - max_lag)
    e = min(L, center + max_lag + 1)

    max_vals = xp_mod.max(xp_mod.abs(corr_norm[:, s:e]), axis=1)

    # Cas dégénérés (signal constant) -> 0.0, comme l'original
    degenerate = (std_x == 0) | (std_y == 0)
    max_vals = xp_mod.where(degenerate, 0.0, max_vals)

    if xp_mod is cp:
        return cp.asnumpy(max_vals)
    return np.asarray(max_vals)


def compute_max_ncc_single(x, y, max_lag=500):
    """Version scalaire (compat. API originale), utilisée ponctuellement."""
    res = compute_max_ncc_batch_gpu(x[None, :], y[None, :], max_lag=max_lag)
    return float(res[0])


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  [GPU] Fréquence dominante — somme de Fourier vectorisée (matricielle)║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# L'original calcule, pour CHAQUE fréquence f d'une grille de 5000 valeurs :
#     F(f) = sum_t  signal[t] * exp(-2j*pi*f*t)
# avec une boucle Python -> O(n_freq * n_pts) en pur Python (très lent).
#
# Sur GPU on remplace ça par UNE SEULE multiplication matricielle :
#     F = exp(-2j*pi * outer(freqs, t)) @ signal
# Et on le fait pour TOUTES les fenêtres du batch en même temps (tenseur 3D
# -> produit batché), donnant un unique kernel GPU au lieu de
# N_fenetres * 5000 itérations Python.

def _time_seconds_from_axis(time_axis):
    """Convertit un axe temporel (datetime ou numérique) en secondes depuis t0."""
    if isinstance(time_axis, pd.Series):
        t = (time_axis - time_axis.iloc[0]).dt.total_seconds().values
    else:
        try:
            time_dt = np.asarray(time_axis, dtype='datetime64[ns]')
            t = (time_dt - time_dt[0]) / np.timedelta64(1, 's')
        except ValueError:
            t = np.asarray(time_axis, dtype=float)
            t = t - t[0]
    return t.astype(np.float64)


def compute_dominant_frequency_batch_gpu(signals, time_axes, n_freq=N_FREQ_BINS,
                                          xp_mod=None, freq_chunk=None):
    """
    Version BATCH de compute_dominant_frequency : calcule la fréquence
    dominante pour N signaux sur GPU.

    ⚠️ IMPORTANT (fix OOM) : on NE matérialise JAMAIS de tensor 3D
    (n_batch, n_freq, L). Chaque fenêtre a sa propre grille de fréquences
    (f_min/f_max dépendent de sa durée/dt), donc le calcul reste un vrai
    produit MATRICIEL 2D par fenêtre :
        F (n_freq,) = M (n_freq, L) @ signal (L,)
    où M = exp(-2j*pi*outer(freqs, t)).
    Mémoire pour une fenêtre : n_freq * L * 16 bytes (complex128).
    Exemple : 5000 * 5000 * 16 = 400 MB -> gérable, contre 100+ GB avant.

    Le "batch" GPU se fait maintenant sur les FRÉQUENCES (freq_chunk) et
    non sur les fenêtres, ce qui borne la mémoire indépendamment du
    nombre de fenêtres traitées.

    signals    : array NumPy (N, L)
    time_axes  : array NumPy (N, L) (datetime64 ou numérique)
    Retourne   : array NumPy (N,) des fréquences dominantes (Hz)
    """
    xp_mod = xp_mod or xp
    N, L = signals.shape

    if freq_chunk is None:
        freq_chunk = n_freq  # par défaut : tout d'un coup (ok si L est petit)

    # Construire les axes temps en secondes (CPU, c'est juste de la conversion)
    t_secs = np.empty((N, L), dtype=np.float64)
    for i in range(N):
        t_secs[i] = _time_seconds_from_axis(time_axes[i])

    sig = signals - signals.mean(axis=1, keepdims=True)

    duration = t_secs[:, -1] - t_secs[:, 0]
    dt_mean = np.mean(np.diff(t_secs, axis=1), axis=1)

    valid = (duration > 0) & (dt_mean > 0)

    f_min = np.where(valid, 1.0 / np.where(duration > 0, duration, 1.0), 0.0)
    f_max = np.where(valid, 0.5 / np.where(dt_mean > 0, dt_mean, 1.0), 0.0)

    dominant_freqs = np.zeros(N, dtype=np.float64)

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return dominant_freqs

    lin_cpu = np.linspace(0.0, 1.0, n_freq)  # (n_freq,) — calculé une seule fois

    # ── Boucle sur les fenêtres valides ─────────────────────────────────
    # (chaque fenêtre a sa propre grille de fréquences -> pas de vrai
    #  batching possible sur cette dimension sans tensor 3D ; mais CHAQUE
    #  fenêtre individuelle est elle-même un produit matriciel GPU massif,
    #  donc le speedup vs boucle Python reste énorme : 5000x moins
    #  d'itérations Python par fenêtre, juste 1 kernel GPU au lieu de 5000)
    for idx in valid_idx:
        t_w = xp_mod.asarray(t_secs[idx], dtype=xp_mod.float64)        # (L,)
        sig_w = xp_mod.asarray(sig[idx], dtype=xp_mod.complex128)      # (L,)
        freqs_w_cpu = f_min[idx] + lin_cpu * (f_max[idx] - f_min[idx])  # (n_freq,) CPU

        best_val = -1.0
        best_freq = 0.0

        # Sous-découpage de la grille de fréquences pour borner la mémoire :
        # taille du bloc M = freq_chunk * L * 16 bytes
        for fstart in range(0, n_freq, freq_chunk):
            fend = min(fstart + freq_chunk, n_freq)
            freqs_block = xp_mod.asarray(freqs_w_cpu[fstart:fend])  # (fb,)

            # M[f, l] = exp(-2j*pi*freqs_block[f]*t_w[l])  -> vrai produit 2D
            phase = xp_mod.outer(freqs_block, t_w)               # (fb, L) réel
            M = xp_mod.exp(-2j * xp_mod.pi * phase)               # (fb, L) complexe

            # F[f] = sum_l signal[l] * M[f, l]  ==  M @ signal
            F_block = M @ sig_w                                   # (fb,)
            spectrum_block = xp_mod.abs(F_block)

            block_best_idx = int(xp_mod.argmax(spectrum_block))
            block_best_val = float(spectrum_block[block_best_idx])

            if block_best_val > best_val:
                best_val = block_best_val
                best_freq = float(freqs_block[block_best_idx])

            del phase, M, F_block, spectrum_block, freqs_block

        dominant_freqs[idx] = best_freq

    return dominant_freqs


def compute_dominant_frequency_single(signal, time_axis, n_freq=N_FREQ_BINS, freq_chunk=None):
    """Version scalaire (compat. API originale)."""
    res = compute_dominant_frequency_batch_gpu(
        np.asarray(signal)[None, :], np.asarray(time_axis, dtype=object)[None, :],
        n_freq=n_freq, freq_chunk=freq_chunk or FREQ_CHUNK
    )
    return float(res[0])


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ [Cell 4 GPU] Groupement par NCC — NCC consécutives pré-calculées      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# L'algorithme de groupement (track_events_temporal) est intrinsèquement
# SÉQUENTIEL : la décision "même événement ou nouveau" dépend de l'état
# précédent (cur["last_window"] change dynamiquement). Impossible de
# paralléliser la boucle elle-même sans changer la sémantique.
#
# MAIS : la majorité du temps n'est PAS passée dans la boucle Python,
# elle est passée dans compute_max_ncc. Et il y a au plus (N-1) calculs
# de NCC nécessaires dans le pire cas (un par transition), donc on PEUT
# pré-calculer toutes les NCC "fenêtre i vs fenêtre i+1" en un seul batch
# GPU. Le souci : l'original compare contre "last_window" qui change de
# valeur selon le résultat précédent (pas toujours i vs i+1 strictement).
#
# Solution pragmatique : on garde la boucle séquentielle (elle est O(N),
# rapide en Python pur), mais on accélère l'opération coûteuse à
# l'intérieur (compute_max_ncc) en la rendant GPU-résidente : les données
# restent sur GPU entre deux appels consécutifs au lieu de faire des
# aller-retours CPU<->GPU à chaque comparaison.

def track_events_temporal_gpu(windows, is_anomaly, timestamps, threshold=0.30, max_lag=500):
    """
    Identique en sémantique à track_events_temporal de l'original,
    mais utilise compute_max_ncc_single (GPU) pour le calcul de similarité.
    """
    events = []
    cur = None

    for i in range(len(is_anomaly)):
        if is_anomaly[i] == 1:
            if cur is None:
                cur = {
                    "debut": timestamps[i],
                    "fin": timestamps[i],
                    "indices": [i],
                    "last_window": windows[i],
                    "sims": []
                }
            else:
                sim = compute_max_ncc_single(cur["last_window"], windows[i], max_lag=max_lag)
                if sim >= threshold:
                    cur["fin"] = timestamps[i]
                    cur["indices"].append(i)
                    cur["last_window"] = windows[i]
                    cur["sims"].append(sim)
                else:
                    events.append(cur)
                    cur = {
                        "debut": timestamps[i],
                        "fin": timestamps[i],
                        "indices": [i],
                        "last_window": windows[i],
                        "sims": []
                    }
        else:
            if cur:
                events.append(cur)
                cur = None

    if cur:
        events.append(cur)

    return events


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║ [Cell 5 GPU] Pré-calcul BATCH des fréquences dominantes + NCC export  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
#
# L'original recalcule compute_dominant_frequency ET compute_max_ncc
# pendant la boucle d'export (une fenêtre à la fois). On les PRÉ-CALCULE
# tous en GPU AVANT la boucle d'export, en un seul (ou quelques) batch(s),
# puis la boucle d'export ne fait plus que du matplotlib (CPU, parallélisable).

def precompute_all_metrics_gpu(anomaly_events, windows, time_arrays):
    """
    Calcule en batch GPU :
      - la fréquence dominante de CHAQUE fenêtre impliquée dans un événement
      - la NCC de chaque fenêtre vs la première fenêtre de son événement
    Retourne deux dicts {win_idx: valeur}.
    """
    all_indices = []
    ref_indices = []  # pour chaque win_idx, l'indice de la fenêtre de référence

    for event in anomaly_events:
        indices = event["indices"]
        ref_idx = indices[0]
        for win_idx in indices:
            all_indices.append(win_idx)
            ref_indices.append(ref_idx)

    all_indices = np.array(all_indices)
    ref_indices = np.array(ref_indices)

    print(f"📊 Pré-calcul GPU : {len(all_indices)} fenêtres "
          f"(fréquence dominante + NCC vs référence)...")

    t0 = time.time()

    # ── Fréquences dominantes en batch ──────────────────────────────────
    sigs = windows[all_indices]
    times = time_arrays[all_indices]
    dom_freqs = compute_dominant_frequency_batch_gpu(
        sigs, times, n_freq=N_FREQ_BINS, freq_chunk=FREQ_CHUNK
    )

    # ── NCC vs référence en batch (sauf les fenêtres qui SONT la référence) ──
    is_ref = (all_indices == ref_indices)
    non_ref_mask = ~is_ref

    ncc_values = np.zeros(len(all_indices), dtype=np.float64)
    if non_ref_mask.any():
        x_batch = windows[ref_indices[non_ref_mask]]
        y_batch = windows[all_indices[non_ref_mask]]
        ncc_values[non_ref_mask] = compute_max_ncc_batch_gpu(x_batch, y_batch, max_lag=NCC_MAX_LAG)

    elapsed = time.time() - t0
    print(f"✅ Pré-calcul terminé en {elapsed:.2f}s "
          f"({len(all_indices)/elapsed:.0f} fenêtres/s)")

    dom_freq_map = dict(zip(all_indices.tolist(), dom_freqs.tolist()))
    ncc_map = dict(zip(all_indices.tolist(), ncc_values.tolist()))
    is_ref_map = dict(zip(all_indices.tolist(), is_ref.tolist()))

    return dom_freq_map, ncc_map, is_ref_map


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║      [Cell 5 GPU] Export des plots — parallélisé en multiprocessing  ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _plot_one_window(args):
    """Worker exécuté dans un process séparé : trace + sauvegarde une fenêtre."""
    (win_idx, num_ev, i_in_ev, signal, time_axis,
     dom_freq, ncc, is_ref, event_dir, folder_name) = args

    freq_str = f"Freq Dom: {dom_freq:.1f}Hz"

    if is_ref:
        score_str = f"REFERENCE_{dom_freq:.1f}Hz"
        score_title = f"(référence | {freq_str})"
    else:
        score_str = f"ncc_{ncc:.4f}_{dom_freq:.1f}Hz"
        score_title = f"NCC = {ncc:.4f} | {freq_str}"

    t_rel = _time_seconds_from_axis(time_axis)
    ts_str = pd.to_datetime(time_axis[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, signal, color='firebrick', alpha=0.85,
            linewidth=0.8, label=f'Signal ({freq_str})')

    ax.set_title(
        f"Événement {num_ev:02d} | Fenêtre #{win_idx} | "
        f"{ts_str} | {score_title}"
    )
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    plt.xticks(rotation=15)
    ax.legend(loc="upper right")
    plt.tight_layout()

    filename = f"window_{win_idx:04d}_{score_str}.png"
    full_img_path = os.path.join(event_dir, filename)
    plt.savefig(full_img_path, dpi=100)
    plt.close(fig)

    ref_image_path = f"{folder_name}/{filename}" if is_ref else None

    return {
        "win_idx": win_idx,
        "num_ev": num_ev,
        "dom_freq": dom_freq,
        "filename": filename,
        "ref_image_path": ref_image_path,
    }


def save_anomaly_event_plots_gpu(anomaly_events, windows, time_arrays,
                                  base_output_dir="outputs",
                                  num_workers=NUM_WORKERS,
                                  dom_freq_map=None, ncc_map=None, is_ref_map=None):
    """
    Version GPU-accélérée : les métriques (NCC, freq dominante) sont
    pré-calculées en BATCH sur GPU avant la boucle, et l'export des
    images est parallélisé sur plusieurs processus CPU.
    Accepte des maps pré-calculées pour éviter de les recalculer.
    """
    if not anomaly_events:
        print("Aucun événement à exporter.")
        return dom_freq_map or {}, ncc_map or {}, is_ref_map or {}

    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"--- Sauvegarde dans : '{os.path.abspath(base_output_dir)}' ---")

    # 1️⃣ Pré-calcul GPU (ou réutilisation des maps fournies)
    if dom_freq_map is None or ncc_map is None or is_ref_map is None:
        dom_freq_map, ncc_map, is_ref_map = precompute_all_metrics_gpu(
            anomaly_events, windows, time_arrays
        )

    # 2️⃣ Préparer les dossiers + arguments des workers
    tasks = []
    event_meta = []

    for num_ev, event in enumerate(anomaly_events):
        indices = event["indices"]

        t_debut_val = time_arrays[indices[0]][0]
        t_fin_val = time_arrays[indices[-1]][-1]

        t_debut_str_clean = pd.to_datetime(t_debut_val).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        t_fin_str_clean = pd.to_datetime(t_fin_val).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        folder_nom_debut = t_debut_str_clean.replace(":", "-").replace(".", "_").replace(" ", "_")
        folder_name = f"evenement_{num_ev:02d}_debut_{folder_nom_debut}"
        event_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(event_dir, exist_ok=True)

        event_meta.append({
            "num_ev": num_ev,
            "folder_name": folder_name,
            "t_debut_str_clean": t_debut_str_clean,
            "t_fin_str_clean": t_fin_str_clean,
            "indices": indices,
        })

        for i_in_ev, win_idx in enumerate(indices):
            if not is_ref_map[win_idx]:
                continue
            tasks.append((
                win_idx, num_ev, i_in_ev,
                windows[win_idx], time_arrays[win_idx],
                dom_freq_map[win_idx], ncc_map[win_idx], is_ref_map[win_idx],
                event_dir, folder_name
            ))

    print(f"🚀 Export de {len(tasks)} plots (fenêtres ref uniquement, {num_workers} workers)...")
    t0 = time.time()

    results_by_event = {num_ev: {"dom_freqs": [], "ref_image_path": None}
                         for num_ev in range(len(anomaly_events))}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_plot_one_window, task) for task in tasks]
        for future in as_completed(futures):
            res = future.result()
            results_by_event[res["num_ev"]]["dom_freqs"].append(res["dom_freq"])
            if res["ref_image_path"]:
                results_by_event[res["num_ev"]]["ref_image_path"] = res["ref_image_path"]

    elapsed = time.time() - t0
    print(f"✅ Export terminé en {elapsed:.2f}s ({len(tasks)/elapsed:.1f} plots/s)")

    # 3️⃣ Construction du récapitulatif CSV + HTML (identique à l'original)
    recap_data = []
    html_content = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport des Anomalies</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f6fa; }",
        "h1 { color: #333; }",
        ".anomaly-card { background: #fff; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".header-info { display: flex; flex-direction: column; gap: 5px; margin-bottom: 15px; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }",
        ".stat { font-weight: bold; color: #e74c3c; }",
        "</style>",
        "</head><body>",
        f"<h1>Rapport d'Analyse : {len(anomaly_events)} événements détectés</h1>"
    ]

    for meta in event_meta:
        num_ev = meta["num_ev"]
        res = results_by_event[num_ev]
        moy_freq = round(np.mean(res["dom_freqs"]), 2) if res["dom_freqs"] else 0.0

        recap_data.append({
            "anomalie_id": f"anomalie {num_ev + 1}",
            "date_de_debut": meta["t_debut_str_clean"],
            "date_de_fin": meta["t_fin_str_clean"],
            "nb_de_fenetres": len(meta["indices"]),
            "moyenne_frequence_dominante_Hz": moy_freq
        })

        html_content.append("<div class='anomaly-card'>")
        html_content.append(f"<h2>Anomalie {num_ev + 1}</h2>")
        html_content.append("<div class='header-info'>")
        html_content.append(f"<span><strong>Date de début :</strong> {meta['t_debut_str_clean']}</span>")
        html_content.append(f"<span><strong>Date de fin :</strong> {meta['t_fin_str_clean']}</span>")
        html_content.append(f"<span><strong>Nombre de fenêtres :</strong> <span class='stat'>{len(meta['indices'])}</span></span>")
        html_content.append(f"<span><strong>Moyenne de fréquence dominante :</strong> <span class='stat'>{moy_freq} Hz</span></span>")
        html_content.append("</div>")
        if res["ref_image_path"]:
            html_content.append("<h3>Plot de la fenêtre de référence :</h3>")
            html_content.append(f"<img src='{res['ref_image_path']}' alt='Reference Window' />")
        html_content.append("</div>")

    html_content.append("</body></html>")

    df_recap = pd.DataFrame(recap_data)
    recap_csv_path = os.path.join(base_output_dir, "recapitulatif_anomalies.csv")
    df_recap.to_csv(recap_csv_path, index=False, sep=";")

    recap_html_path = os.path.join(base_output_dir, "rapport_anomalies.html")
    with open(recap_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"\n--- Export terminé : {len(anomaly_events)} événements sauvegardés ---")
    print(f"--- Récapitulatif CSV : '{recap_csv_path}' ---")
    print(f"--- Rapport HTML : '{recap_html_path}' ---")

    return dom_freq_map, ncc_map, is_ref_map


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║          CLUSTERING INTER-ÉVÉNEMENTS (SUPER-TYPES)                    ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def cluster_events_by_type(anomaly_events, windows):
    """
    2e niveau de clustering : NCC entre fenêtres de référence de tous les events.
    Deux events temporellement disjoints mais de même forme → même type.
    Retourne (labels: list[int], ncc_matrix: np.ndarray (N×N)).
    """
    N = len(anomaly_events)
    if N == 0:
        return [], np.zeros((0, 0))
    if N == 1:
        return [0], np.ones((1, 1))

    ref_indices = np.array([ev['indices'][0] for ev in anomaly_events])
    ref_windows = windows[ref_indices]  # (N, 5000)

    # Toutes les paires i<j en un batch chunké
    pairs_i, pairs_j = np.triu_indices(N, k=1)
    ncc_flat = np.zeros(len(pairs_i), dtype=np.float64)

    for start in range(0, len(pairs_i), GPU_BATCH_SIZE):
        end = min(start + GPU_BATCH_SIZE, len(pairs_i))
        ncc_flat[start:end] = compute_max_ncc_batch_gpu(
            ref_windows[pairs_i[start:end]],
            ref_windows[pairs_j[start:end]],
            max_lag=NCC_MAX_LAG
        )

    ncc_matrix = np.eye(N, dtype=np.float64)
    ncc_matrix[pairs_i, pairs_j] = ncc_flat
    ncc_matrix[pairs_j, pairs_i] = ncc_flat

    dist_condensed = squareform(np.clip(1.0 - ncc_matrix, 0.0, 1.0), checks=False)
    Z = linkage(dist_condensed, method='average')
    raw_labels = fcluster(Z, t=1.0 - NCC_TYPE_THRESHOLD, criterion='distance')

    zero_labels = (raw_labels - 1).tolist()

    # Renumérotation par ordre de première apparition (Type 0, 1, 2, …)
    seen: dict = {}
    next_id = 0
    remapped = []
    for lbl in zero_labels:
        if lbl not in seen:
            seen[lbl] = next_id
            next_id += 1
        remapped.append(seen[lbl])

    return remapped, ncc_matrix


def build_signal_profile(anomaly_events, windows, time_arrays,
                          dom_freq_map, ncc_map, cluster_labels, signal_id="signal"):
    """
    Construit un dict structuré (SignalProfile) à partir des sorties du pipeline.
    Contient toutes les métadonnées nécessaires pour la corrélation et les exports.
    """
    events_out = []
    for ev_idx, event in enumerate(anomaly_events):
        indices   = event['indices']
        ref_idx   = indices[0]
        win_ts    = [pd.to_datetime(time_arrays[i][0]) for i in indices]
        debut_ts  = pd.to_datetime(event['debut'])
        fin_ts    = pd.to_datetime(event['fin'])
        duration  = max((fin_ts - debut_ts).total_seconds(), 0.0)

        events_out.append({
            'event_id':          ev_idx,
            'cluster_id':        cluster_labels[ev_idx],
            'ref_win_idx':       ref_idx,
            'ref_timestamp':     win_ts[0],
            'ref_window':        windows[ref_idx],
            'ref_dom_freq':      float(dom_freq_map.get(ref_idx, 0.0)),
            'window_timestamps': win_ts,
            'ncc_vs_ref':        [float(ncc_map.get(i, 1.0)) for i in indices],
            'dom_freqs':         [float(dom_freq_map.get(i, 0.0)) for i in indices],
            'window_count':      len(indices),
            'duration_s':        duration,
        })

    t_start = min(e['ref_timestamp'] for e in events_out) if events_out else None
    t_end   = max(e['window_timestamps'][-1] for e in events_out) if events_out else None
    n_clust = max(cluster_labels, default=-1) + 1

    return {
        'signal_id':     signal_id,
        'events':        events_out,
        'total_windows': sum(e['window_count'] for e in events_out),
        't_start':       t_start,
        't_end':         t_end,
        'n_clusters':    n_clust,
    }


def plot_global_timeline(signal_profile, output_dir):
    """
    Vue Gantt : un rectangle par événement, coloré par type.
    Permet de voir d'un coup d'œil la distribution temporelle de chaque type.
    """
    events = signal_profile['events']
    if not events:
        return None

    n_clust = signal_profile['n_clusters']
    cmap    = plt.cm.get_cmap('Set2', max(n_clust, 1))

    fig, ax = plt.subplots(figsize=(18, max(4, len(events) * 0.22 + 2)))

    for ev in events:
        t_left   = mdates.date2num(ev['ref_timestamp'])
        dur_days = max(ev['duration_s'] / 86400.0, 1.0 / 86400.0)
        ax.barh(ev['event_id'], dur_days, left=t_left, height=0.7,
                color=cmap(ev['cluster_id']), alpha=0.85)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.set_xlabel("Heure")
    ax.set_ylabel("Événement #")
    ax.set_title(
        f"Timeline globale — {signal_profile['signal_id']}\n"
        f"{len(events)} événements  |  {n_clust} types  |  "
        f"{signal_profile['total_windows']} fenêtres totales"
    )

    seen, handles = set(), []
    for ev in events:
        cid = ev['cluster_id']
        if cid not in seen:
            seen.add(cid)
            handles.append(mpatches.Patch(color=cmap(cid), label=f"Type {cid}"))
    ax.legend(handles=sorted(handles, key=lambda h: int(h.get_label().split()[-1])),
              loc='upper right', bbox_to_anchor=(1.12, 1))

    plt.tight_layout()
    path = os.path.join(output_dir, "timeline_globale.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_type_histogram_worker(args):
    """Worker : génère et sauvegarde l'histogramme temporel d'un type d'anomalie."""
    (cid, window_timestamps_flat, dom_freqs_flat,
     t_start, t_end, bin_width_min, n_events, types_dir) = args

    all_ts    = pd.DatetimeIndex(window_timestamps_flat)
    freq      = f'{bin_width_min}min'
    counts    = all_ts.floor(freq).value_counts().sort_index()
    full_bins = pd.date_range(t_start.floor(freq), t_end.ceil(freq), freq=freq)
    counts    = counts.reindex(full_bins, fill_value=0)

    mean_c  = counts.mean()
    std_c   = counts.std()
    burst_t = mean_c + 2.0 * std_c

    bin_centers = counts.index + pd.Timedelta(minutes=bin_width_min / 2)
    bar_colors  = ['#e74c3c' if v > burst_t else '#3498db' for v in counts.values]

    total_w     = int(counts.sum())
    active_bins = int((counts > 0).sum())
    recur_rate  = active_bins / max(len(counts), 1)
    p = counts.values / total_w if total_w > 0 else counts.values
    p = p[p > 0]
    entropy   = (-np.sum(p * np.log2(p)) / np.log2(max(len(counts), 2))
                 if len(p) > 0 else 0.0)
    peak_time = (counts.idxmax() + pd.Timedelta(minutes=bin_width_min / 2)
                 ).strftime('%H:%M') if total_w > 0 else 'N/A'
    med_freq  = float(np.median(dom_freqs_flat)) if dom_freqs_flat else 0.0

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(mdates.date2num(bin_centers), counts.values,
           width=bin_width_min / 1440.0,
           color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.4)
    if burst_t > 0:
        ax.axhline(burst_t, color='#e74c3c', linestyle='--', linewidth=1.2,
                   label=f'Seuil burst (μ+2σ = {burst_t:.1f})')

    ax.set_title(
        f"Type {cid}  |  {n_events} événements  |  {total_w} fenêtres  |  "
        f"Fréq. dominante médiane : {med_freq:.1f} Hz\n"
        f"Récurrence : {recur_rate:.0%}  |  Pic : {peak_time}  |  "
        f"Entropie temporelle : {entropy:.2f}"
    )
    ax.set_xlabel("Heure d'apparition")
    ax.set_ylabel("Nombre de fenêtres")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    if burst_t > 0:
        ax.legend()

    plt.tight_layout()
    path = os.path.join(types_dir, f"type_{cid:02d}_histogram.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return cid, path


def plot_type_histograms(signal_profile, output_dir, bin_width_min=HIST_BIN_MIN):
    """
    Pour chaque type d'anomalie : histogramme temporel (x=heure, y=nb fenêtres).
    Les bins en rouge dépassent le seuil burst (moyenne + 2σ).
    Parallélisé par type via ProcessPoolExecutor.
    """
    events  = signal_profile['events']
    t_start = signal_profile['t_start']
    t_end   = signal_profile['t_end']
    if not events or t_start is None:
        return {}

    types_dir = os.path.join(output_dir, "types")
    os.makedirs(types_dir, exist_ok=True)

    by_cluster = defaultdict(list)
    for ev in events:
        by_cluster[ev['cluster_id']].append(ev)

    tasks = []
    for cid, c_events in sorted(by_cluster.items()):
        tasks.append((
            cid,
            [t for ev in c_events for t in ev['window_timestamps']],
            [f for ev in c_events for f in ev['dom_freqs']],
            t_start, t_end, bin_width_min, len(c_events), types_dir
        ))

    histogram_paths = {}
    n_workers = min(NUM_WORKERS, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for cid, path in executor.map(_plot_type_histogram_worker, tasks):
            histogram_paths[cid] = path

    return histogram_paths


def plot_ncc_matrix(ncc_matrix, cluster_labels, output_dir):
    """
    Heatmap NCC entre toutes les fenêtres de référence, triée par type.
    Skippée si trop d'événements (illisible au-delà de MAX_EVENTS_NCC_MATRIX).
    """
    N = len(cluster_labels)
    if N <= 1:
        return None
    if N > MAX_EVENTS_NCC_MATRIX:
        print(f"  Matrice NCC ignorée ({N} > {MAX_EVENTS_NCC_MATRIX} événements — illisible)")
        return None

    order        = np.argsort(cluster_labels)
    ncc_sorted   = ncc_matrix[np.ix_(order, order)]
    labels_sorted = np.array(cluster_labels)[order]

    cell_size = max(0.35, min(0.6, 8.0 / N))
    fig_size  = max(6, N * cell_size)
    fig, ax   = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(ncc_sorted, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, ax=ax, label='NCC', fraction=0.046, pad=0.04)

    # Frontières entre clusters
    boundaries = [0]
    prev = labels_sorted[0]
    for i, lbl in enumerate(labels_sorted[1:], 1):
        if lbl != prev:
            boundaries.append(i)
            prev = lbl
    boundaries.append(N)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='black', linewidth=1.5)
        ax.axvline(b - 0.5, color='black', linewidth=1.5)

    ax.set_title(f"Matrice NCC inter-références ({N} événements, triée par type)")
    ax.set_xlabel("Événement #")
    ax.set_ylabel("Événement #")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(output_dir, "ncc_matrix_events.png")
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_ref_window_worker(args):
    """Worker : génère et sauvegarde le plot de référence d'un événement."""
    ev_id, cid, signal, time_ax, dom_freq, refs_dir, output_dir = args

    t_rel  = _time_seconds_from_axis(time_ax)
    ts_str = pd.to_datetime(time_ax[0]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rel, signal, color='firebrick', alpha=0.85, linewidth=0.8)
    ax.set_title(
        f"Événement {ev_id:02d} — Type {cid} — Fenêtre de référence\n"
        f"{ts_str}  |  Fréq. dominante : {dom_freq:.1f} Hz"
    )
    ax.set_xlabel("Temps relatif (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()

    fname = f"event_{ev_id:02d}_type{cid}_ref.png"
    fpath = os.path.join(refs_dir, fname)
    fig.savefig(fpath, dpi=100)
    plt.close(fig)
    return ev_id, os.path.relpath(fpath, output_dir)


def plot_reference_windows(signal_profile, time_arrays, output_dir):
    """
    Génère un plot du signal de référence pour chaque événement.
    Parallélisé via ProcessPoolExecutor (un plot par event).
    Retourne {event_id: rel_path_from_output_dir}.
    """
    events = signal_profile['events']
    if not events:
        return {}

    refs_dir = os.path.join(output_dir, "refs")
    os.makedirs(refs_dir, exist_ok=True)

    tasks = [
        (ev['event_id'], ev['cluster_id'], ev['ref_window'],
         time_arrays[ev['ref_win_idx']], ev['ref_dom_freq'],
         refs_dir, output_dir)
        for ev in events
    ]

    ref_paths = {}
    n_workers = min(NUM_WORKERS, len(tasks))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for ev_id, rel_path in executor.map(_plot_ref_window_worker, tasks):
            ref_paths[ev_id] = rel_path

    print(f"  -> {len(ref_paths)} plots de référence générés dans 'refs/'")
    return ref_paths


def export_enriched_html(signal_profile, base_output_dir,
                          histogram_paths, timeline_path, ncc_matrix_path,
                          ref_plot_paths=None):
    """
    Rapport HTML enrichi : timeline + types + histogrammes + matrice NCC + tableau détail.
    """
    events    = signal_profile['events']
    n_clust   = signal_profile['n_clusters']
    by_cluster = defaultdict(list)
    for ev in events:
        by_cluster[ev['cluster_id']].append(ev)

    def rel(p):
        return os.path.relpath(p, base_output_dir) if p else None

    html = [
        "<html><head><meta charset='utf-8'>",
        "<title>Rapport Enrichi — Analyse des Anomalies</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f5f6fa}",
        "h1,h2,h3{color:#2c3e50}",
        ".card{background:#fff;margin-bottom:24px;padding:20px;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,.12)}",
        ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:20px}",
        ".kpi{background:#2c3e50;color:#fff;padding:16px;border-radius:8px;text-align:center}",
        ".kpi .val{font-size:28px;font-weight:bold;color:#e74c3c}",
        ".kpi .lbl{font-size:12px;margin-top:4px}",
        ".badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:#3498db;color:#fff;margin:2px}",
        "img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;margin-top:8px}",
        "table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ddd;padding:8px 12px;text-align:left;font-size:13px}",
        "th{background:#2c3e50;color:#fff}",
        "tr:nth-child(even){background:#f9f9f9}",
        "</style></head><body>",
        f"<h1>Rapport d'Analyse Enrichi — {signal_profile['signal_id']}</h1>",
    ]

    # KPIs
    html += [
        "<div class='card'><div class='kpi-grid'>",
        f"<div class='kpi'><div class='val'>{len(events)}</div><div class='lbl'>Événements</div></div>",
        f"<div class='kpi'><div class='val'>{n_clust}</div><div class='lbl'>Types distincts</div></div>",
        f"<div class='kpi'><div class='val'>{signal_profile['total_windows']}</div><div class='lbl'>Fenêtres totales</div></div>",
        "</div></div>",
    ]

    # 1. Timeline
    html += ["<div class='card'>", "<h2>1. Timeline globale</h2>"]
    if timeline_path:
        html.append(f"<img src='{rel(timeline_path)}' alt='Timeline globale'/>")
    html.append("</div>")

    # 2. Tableau récapitulatif par type
    html += ["<div class='card'>", "<h2>2. Synthèse par type</h2>",
             "<table>",
             "<tr><th>Type</th><th>Événements</th><th>Fenêtres</th>"
             "<th>Fréq. dominante médiane (Hz)</th><th>Durée moy. (s)</th></tr>"]
    for cid in sorted(by_cluster):
        c_ev   = by_cluster[cid]
        tot_w  = sum(e['window_count'] for e in c_ev)
        freqs  = [f for e in c_ev for f in e['dom_freqs']]
        mf     = np.median(freqs) if freqs else 0.0
        md     = np.mean([e['duration_s'] for e in c_ev])
        html.append(f"<tr><td><strong>Type {cid}</strong></td>"
                    f"<td>{len(c_ev)}</td><td>{tot_w}</td>"
                    f"<td>{mf:.1f}</td><td>{md:.1f}</td></tr>")
    html += ["</table>", "</div>"]

    # 3. Histogrammes
    html += ["<div class='card'>", "<h2>3. Répartition temporelle par type</h2>"]
    for cid in sorted(histogram_paths):
        html.append(f"<h3>Type {cid} — {len(by_cluster[cid])} événements</h3>")
        html.append(f"<img src='{rel(histogram_paths[cid])}' alt='Histogramme type {cid}'/>")
    html.append("</div>")

    # 4. Matrice NCC
    if ncc_matrix_path:
        html += ["<div class='card'>", "<h2>4. Matrice de similarité inter-références</h2>",
                 f"<img src='{rel(ncc_matrix_path)}' alt='Matrice NCC'/>",
                 "</div>"]

    # 5. Tableau détail événements
    html += ["<div class='card'>", "<h2>5. Détail des événements</h2>",
             "<table>",
             "<tr><th>#</th><th>Type</th><th>Début</th><th>Fenêtres</th>"
             "<th>Durée (s)</th><th>Fréq. moy. (Hz)</th><th>NCC moy. vs réf.</th></tr>"]
    for ev in events:
        mf  = np.mean(ev['dom_freqs']) if ev['dom_freqs'] else 0.0
        mn  = np.mean(ev['ncc_vs_ref']) if ev['ncc_vs_ref'] else 1.0
        ts  = ev['ref_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        html.append(
            f"<tr><td>{ev['event_id']}</td>"
            f"<td><span class='badge'>Type {ev['cluster_id']}</span></td>"
            f"<td>{ts}</td><td>{ev['window_count']}</td>"
            f"<td>{ev['duration_s']:.1f}</td><td>{mf:.1f}</td><td>{mn:.4f}</td></tr>"
        )
    html += ["</table>", "</div>"]

    # 6. Fenêtres de référence par événement
    if ref_plot_paths:
        html += ["<div class='card'>",
                 "<h2>6. Fenêtres de référence par événement</h2>"]
        for cid in sorted(by_cluster):
            c_events = sorted(by_cluster[cid], key=lambda e: e['event_id'])
            html.append(
                f"<h3>Type {cid} &mdash; {len(c_events)} événement(s)</h3>"
            )
            html.append(
                "<div style='display:grid;grid-template-columns:"
                "repeat(auto-fit,minmax(480px,1fr));gap:16px'>"
            )
            for ev in c_events:
                ev_id = ev['event_id']
                if ev_id not in ref_plot_paths:
                    continue
                ts = ev['ref_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                html.append(
                    f"<div style='border:1px solid #ddd;border-radius:6px;"
                    f"padding:12px;background:#fafafa'>"
                    f"<p style='margin:0 0 6px'>"
                    f"<strong>Événement {ev_id}</strong> &bull; {ts} &bull; "
                    f"{ev['window_count']} fen&ecirc;tres &bull; "
                    f"Dur&eacute;e&nbsp;: {ev['duration_s']:.1f}&nbsp;s &bull; "
                    f"Fr&eacute;q.&nbsp;r&eacute;f.&nbsp;: {ev['ref_dom_freq']:.1f}&nbsp;Hz</p>"
                    f"<img src='{ref_plot_paths[ev_id]}' "
                    f"alt='R&eacute;f&eacute;rence &eacute;v&eacute;nement {ev_id}'/>"
                    f"</div>"
                )
            html.append("</div>")  # close grid
        html.append("</div>")  # close card

    html += ["</body></html>"]

    path = os.path.join(base_output_dir, "rapport_enrichi.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f"--- Rapport enrichi : '{path}' ---")
    return path


def serialize_signal_profile(signal_profile, output_dir):
    """Sérialise le profil (sans les arrays numpy) en JSON pour oscillo_correlation.py."""
    def ts(t):
        return pd.Timestamp(t).isoformat() if t is not None else None

    data = {
        'signal_id':     signal_profile['signal_id'],
        'total_windows': signal_profile['total_windows'],
        'n_clusters':    signal_profile['n_clusters'],
        't_start':       ts(signal_profile['t_start']),
        't_end':         ts(signal_profile['t_end']),
        'events': [
            {
                'event_id':          ev['event_id'],
                'cluster_id':        ev['cluster_id'],
                'ref_timestamp':     ts(ev['ref_timestamp']),
                'ref_dom_freq':      ev['ref_dom_freq'],
                'window_count':      ev['window_count'],
                'duration_s':        ev['duration_s'],
                'window_timestamps': [ts(t) for t in ev['window_timestamps']],
                'dom_freqs':         ev['dom_freqs'],
                'ncc_vs_ref':        ev['ncc_vs_ref'],
            }
            for ev in signal_profile['events']
        ],
    }

    path = os.path.join(output_dir, "signal_profile.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"--- Profil JSON sérialisé : '{path}' ---")
    return path


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                     ║
# ╚═══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":

    t_start_total = time.time()

    # ── Résolution du dossier de sortie ─────────────────────────────────
    if USE_DATALAKE:
        output_dir = (f"outputs_{DATALAKE_BOITIER}_v{DATALAKE_VOIE}_{DATALAKE_DATE}")
        signal_id  = f"{DATALAKE_BOITIER}/voie_{DATALAKE_VOIE}/{DATALAKE_DATE}"
    else:
        output_dir = OUTPUT_DIR
        signal_id  = os.path.basename(os.path.normpath(DATA_PATH))

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PIPELINE GPU — Groupement de Fenêtres d'Anomalies")
    print(f"   Source  : {'Data Lake Parquet' if USE_DATALAKE else 'Fichiers bruts'}")
    if USE_DATALAKE:
        print(f"   Signal  : {signal_id}")
    print(f"   Device  : {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy fallback)'}")
    print(f"   Sortie  : {os.path.abspath(output_dir)}")
    print("=" * 70 + "\n")

    # ── Chargement ───────────────────────────────────────────────────────
    t0 = time.time()
    if USE_DATALAKE:
        windows, time_arrays = load_from_datalake(
            DATA_LAKE_PATH, DATALAKE_BOITIER, DATALAKE_VOIE, DATALAKE_DATE, TARGET_PTS
        )
    else:
        windows, time_arrays = load_all_oscillo_files(DATA_PATH, CHANNEL_INDEX, TARGET_PTS)
    print(f"⏱️  Chargement : {time.time()-t0:.2f}s")

    if len(windows) > 0:
        timestamps = [ta[0] for ta in time_arrays]
        print(f"Timestamps OK — exemple : {pd.Timestamp(timestamps[0])}")
    else:
        timestamps = []
        print("AVERTISSEMENT : aucune fenêtre chargée.")

    # ── [Cell 3] Toutes les fenêtres = anomalies ────────────────────────
    if len(windows) > 0:
        is_anomaly = np.ones(len(windows), dtype=int)
        print(f"Fenêtres déclarées anomalies : {np.sum(is_anomaly)} / {len(is_anomaly)}")
    else:
        is_anomaly = np.array([], dtype=int)

    # ── [Cell 4] Groupement temporel par NCC (GPU) ──────────────────────
    t0 = time.time()
    if len(is_anomaly) > 0:
        anomaly_events = track_events_temporal_gpu(
            windows, is_anomaly, timestamps,
            threshold=NCC_THRESHOLD, max_lag=NCC_MAX_LAG
        )
        print(f"\nTotal groupes (événements) : {len(anomaly_events)}")
    else:
        anomaly_events = []
        print("Aucune fenêtre à traiter.")
    print(f"⏱️  Groupement NCC : {time.time()-t0:.2f}s")

    # ── Pré-calcul GPU des métriques ─────────────────────────────────────
    dom_freq_map, ncc_map, is_ref_map = {}, {}, {}
    t0 = time.time()
    if len(anomaly_events) > 0:
        dom_freq_map, ncc_map, is_ref_map = precompute_all_metrics_gpu(
            anomaly_events, windows, time_arrays
        )
    print(f"⏱️  Pré-calcul métriques : {time.time()-t0:.2f}s")

    # ── Export plots désactivé (trop coûteux en RAM) ─────────────────────
    print("ℹ️  Export plots désactivé.")

    # ── Clustering par types + visualisations enrichies ──────────────────
    if len(anomaly_events) > 0:
        t0 = time.time()
        print("\n--- Clustering inter-événements (super-types) ---")
        cluster_labels, ncc_matrix = cluster_events_by_type(anomaly_events, windows)
        n_types = max(cluster_labels, default=-1) + 1
        print(f"  -> {len(anomaly_events)} événements → {n_types} types "
              f"(seuil NCC_TYPE={NCC_TYPE_THRESHOLD})")

        signal_profile = build_signal_profile(
            anomaly_events, windows, time_arrays,
            dom_freq_map, ncc_map, cluster_labels,
            signal_id=signal_id
        )

        print("--- Génération des visualisations enrichies ---")
        timeline_path   = plot_global_timeline(signal_profile, output_dir)
        histogram_paths = plot_type_histograms(signal_profile, output_dir)
        ncc_matrix_path = plot_ncc_matrix(ncc_matrix, cluster_labels, output_dir)
        ref_plot_paths  = plot_reference_windows(signal_profile, time_arrays, output_dir)

        export_enriched_html(signal_profile, output_dir,
                              histogram_paths, timeline_path, ncc_matrix_path,
                              ref_plot_paths=ref_plot_paths)
        serialize_signal_profile(signal_profile, output_dir)
        print(f"⏱️  Analyse enrichie : {time.time()-t0:.2f}s")

    print("\n" + "=" * 70)
    print(f"✅ PIPELINE TERMINÉ EN {time.time()-t_start_total:.2f}s")
    print("=" * 70 + "\n")
