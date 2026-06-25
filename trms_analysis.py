# ==============================================================================
# 1. IMPORTS ET CONFIGURATION
# ==============================================================================
import os
import shutil
import glob
import re
import logging
from datetime import timedelta

# --- MULTI-ACCÉLÉRATION : GPU POUR LE RESTE, CPU POUR LES RUPTURES ---
import cudf          # GPU - Chargement rapide et Pics
import cupy as cp    # GPU - Calculs matriciels rapides
from cuml.cluster import KMeans as cumlKMeans  # GPU - Clustering rapide Bloc 2 (Optionnel, mais on va repasser sur Sklearn pour cohérence CPU)

import pandas as pd  # CPU - Détection des ruptures
import numpy as np   # CPU - Détection des ruptures
from sklearn.cluster import KMeans as sklearnKMeans
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import median_filter
import ruptures as rpt

# Configuration de base
DATA_DIR = "/home/aazzouzi/Projet IA/Adapte2/data_normandie/Lot2_Boitier1"

OUTPUT_DIR = "outputs"
OSCILLO_DIR = os.path.join(DATA_DIR, "Oscillo")
DATA_FILE =os.path.join(DATA_DIR, "TRMS", "Courant_Tension_FctTMS_Lot2Boitier1_2025-10-06_17-43_post_traitement.txt")

NUM_VOIES = 1

# Seuils et Paramètres
PEAK_MAX_MIN_DIFF_THRESHOLD = 100
PEAK_MIN_MAX_THRESHOLD = 20
PEAK_EFFICACE_THRESHOLD = 20

WINDOW_SIZE_PEAK = 3600

# Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('Pipeline_IA_Hybride')

COLUMN_NAMES = [
    "time",
    "min1", "max1", "moy1", "efficace1",
    "min2", "max2", "moy2", "efficace2",
    "min3", "max3", "moy3", "efficace3"
]

# ==============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES (MÉMOIRE GPU)
# ==============================================================================
def create_output_dirs(base_dir=OUTPUT_DIR, num_voies=NUM_VOIES):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    dirs = {}
    for v in range(1, NUM_VOIES + 1):
        pics_dir = os.path.join(base_dir, f"anomalie_voie_{v}", "anomalies_pics")
        rupt_dir = os.path.join(base_dir, f"anomalie_voie_{v}", "anomalies_ruptures")

        os.makedirs(pics_dir, exist_ok=True)
        os.makedirs(rupt_dir, exist_ok=True)

        dirs[v] = {"pics": pics_dir, "ruptures": rupt_dir}

    return dirs

def load_data(filepath):
    """Charge en mémoire GPU avec l'astuce de nettoyage pour le bug to_datetime."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    logger.info("Chargement de '%s' sur le GPU...", filepath)
    df = cudf.read_csv(filepath, header=None, names=COLUMN_NAMES, sep=",")

    try:
        df["time"] = cudf.to_datetime(df["time"])
    except Exception:
        logger.warning("Format de date hétérogène détecté. Application du filtre de secours...")
        time_cpu = pd.to_datetime(df["time"].to_pandas(), errors="coerce")
        df["time"] = cudf.Series(time_cpu)

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    
    for col in COLUMN_NAMES[1:]:
        df[col] = cudf.to_numeric(df[col], errors="coerce")

    total_nan = df.isna().sum().sum()
    if total_nan > 0:
        df = df.fillna(0)

    logger.info("Données chargées : %d lignes", len(df))
    return df

# ==============================================================================
# 3. FONCTIONS DE DÉTECTION (BLOC 1 GPU / BLOC 2 SÉCURISÉ CPU)
# ==============================================================================
def segment_signal(df, window, overlap):
    step = window - overlap
    n = len(df)
    segments = []
    idx = 0
    start = 0

    while start + window <= n:
        seg = df.iloc[start: start + window].copy()
        segments.append((idx, seg))
        start += step
        idx += 1

    if start < n and (n - start) > window // 2:
        seg = df.iloc[start:].copy()
        segments.append((idx, seg))

    return segments

def detect_peaks(segment: cudf.DataFrame, voie: int):
    """BLOC 1 : Reste sur le GPU (Parallélisation parfaite via CuPy)."""
    col_min = f"min{voie}"
    col_max = f"max{voie}"
    col_eff = f"efficace{voie}"
    
    cond = ((segment[col_max].abs() - segment[col_min].abs()).abs() > PEAK_MAX_MIN_DIFF_THRESHOLD) | \
        (segment[col_eff] > PEAK_EFFICACE_THRESHOLD)
           
    if not cond.any():
        return None
        
    # Calcul de la Series de valeurs max
    max_vals = segment[col_min].abs() + segment[col_max].abs()
    
    # --- LA CORRECTION ROBUSTE AVEC CUPY ---
    # 1. On convertit la Series cuDF en tableau CuPy (ça reste à 100% en VRAM GPU)
    max_vals_gpu_array = cp.asarray(max_vals)
    
    # 2. On trouve l'index LOCAL au segment avec cp.argmax()
    idx_local = int(cp.argmax(max_vals_gpu_array))
    
    # 3. Comme plot_peak_anomaly utilise .iloc sur le segment, 
    # l'index local (0 à 3600) est EXACTEMENT ce qu'il faut !
    return idx_local


def detect_ruptures_cpu(df_pandas: pd.DataFrame, voie: int):
    """BLOC 2 : Exécuté sur CPU (Évite l'engorgement de la boucle for sur le GPU)."""
    col_eff = f"efficace{voie}"
    signal = df_pandas[col_eff].values.astype(float)
    x = signal
    n = len(x)
    
    smoothing_width = 11
    cost_width = min(5000, max(10, n // 6))
    search_width = 20000

    # Utilisation native de NumPy et SciPy (CPU)
    x_med = median_filter(x, size=smoothing_width, mode="nearest")
    D = np.zeros(len(x_med))

    # Cette boucle tourne maintenant à sa vitesse maximale sur CPU
    for k in range(cost_width, len(x_med) - cost_width):
        median_left = np.median(x_med[k-cost_width:k])
        median_right = np.median(x_med[k:k+cost_width])
        D[k] = median_right - median_left

    D_abs = np.abs(D)
    breakpoints = []
    scores = []

    for start in range(0, len(D_abs), search_width):
        end = min(start + search_width, len(D_abs))
        seg = D_abs[start:end]
        if len(seg) == 0:
            continue
            
        local_idx = np.argmax(seg)
        global_idx = start + local_idx
        
        breakpoints.append(global_idx)
        scores.append(D_abs[global_idx])

    if len(breakpoints) < 2:
        return []

    breakpoints = np.array(breakpoints)
    scores = np.array(scores)
    X = scores.reshape(-1, 1)

    try:
        # Scikit-Learn KMeans (CPU)
        kmeans = sklearnKMeans(n_clusters=2, random_state=0, n_init=20)
        labels = kmeans.fit_predict(X)

        centers = kmeans.cluster_centers_.flatten()
        true_cluster = np.argmax(centers)
        logger.info("  -> Cluster majoritaire (ruptures CPU) : %d", true_cluster)
        true_breaks = breakpoints[labels == true_cluster]
        
        # --- AJUSTEMENT EXACT AVEC BINSEG ---
        exact_breaks = []
        window_size = 2000
        
        for bp in true_breaks:
            start_idx = max(0, bp - window_size)
            end_idx = min(n, bp + window_size)
            sub_signal = x[start_idx:end_idx]
            
            if len(sub_signal) < 10:
                exact_breaks.append(bp)
                continue
                
            try:
                algo = rpt.Binseg(model="l2").fit(sub_signal)
                local_bkps = algo.predict(n_bkps=1)
                
                if len(local_bkps) > 0 and local_bkps[0] < len(sub_signal):
                    exact_breaks.append(start_idx + local_bkps[0])
                else:
                    exact_breaks.append(bp)
            except:
                exact_breaks.append(bp)
                
        return sorted(list(set(exact_breaks)))
    except Exception as e:
        logger.error("Erreur KMeans / Binseg CPU voie %d : %s", voie, e)
        return []


def extract_anomaly_timestamp(filename):
    match = re.search(r"timestamp_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.png", filename)
    return pd.to_datetime(match.group(1), format="%Y-%m-%d_%H-%M-%S") if match else None

def extract_oscillo_starttime(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.txt", filename)
    return pd.to_datetime(match.group(1), format="%Y-%m-%d_%H-%M") if match else None

def find_oscillo_matches(base_output_dir, oscillo_data_dir):
    oscillo_files = glob.glob(os.path.join(oscillo_data_dir, "*.txt"))
    osc_catalog = []
    for osc_path in oscillo_files:
        start_t = extract_oscillo_starttime(os.path.basename(osc_path))
        if start_t:
            osc_catalog.append({"path": osc_path, "start": start_t, "end": start_t + timedelta(minutes=30)})
            
    matches = {}
    for voie in range(1, NUM_VOIES + 1):
        pics_dir = os.path.join(base_output_dir, f"anomalie_voie_{voie}", "anomalies_pics")
        if not os.path.exists(pics_dir): continue
            
        anomalies = glob.glob(os.path.join(pics_dir, "*", "*.png"))
        for anom_path in anomalies:
            if "oscillo" in anom_path: continue
            ts = extract_anomaly_timestamp(os.path.basename(anom_path))
            if ts:
                for osc in osc_catalog:
                    if osc["start"] <= ts < osc["end"]:
                        if osc["path"] not in matches: matches[osc["path"]] = []
                        matches[osc["path"]].append({"voie": voie, "path": anom_path, "ts": ts})
                        break
    return matches

# ==============================================================================
# 4. FONCTIONS DE VISUALISATION (MATPLOTLIB S'EXECUTE SUR CPU)
# ==============================================================================
def plot_peak_anomaly(segment, voie: int, peak_idx_local: int, anomalie_id: int, save_dir: str):
    if isinstance(segment, cudf.DataFrame):
        segment = segment.to_pandas()

    if "time" in segment.columns and pd.notna(segment.iloc[peak_idx_local]["time"]):
        peak_time = segment.iloc[peak_idx_local]["time"]
        timestamp_str = peak_time.strftime("%Y-%m-%d_%H-%M-%S")
        display_time = peak_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = f"index{peak_idx_local}"
        display_time = f"Index {peak_idx_local}"

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Voie {voie} - Plus grand pic détecté à : {display_time}", fontsize=14, fontweight="bold")

    cols = [f"efficace{voie}", f"min{voie}", f"max{voie}"]
    colors = ["#FF9800", "#2196F3", "#F44336"]
    labels = ["Efficace", "Minimum", "Maximum"]
    thresholds = [(PEAK_EFFICACE_THRESHOLD, None), (-PEAK_MIN_MAX_THRESHOLD, PEAK_MIN_MAX_THRESHOLD), (-PEAK_MIN_MAX_THRESHOLD, PEAK_MIN_MAX_THRESHOLD)]

    time_axis = segment["time"].values if "time" in segment.columns else np.arange(len(segment))
    peak_t = segment.iloc[peak_idx_local]["time"] if "time" in segment.columns else time_axis[peak_idx_local]

    for ax, col, color, label, thresh in zip(axes, cols, colors, labels, thresholds):
        ax.plot(time_axis, segment[col].values, color=color, linewidth=1.5, label=label)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if thresh is not None:
            if thresh[0] is not None: ax.axhline(thresh[0], color="gray", linestyle="--", linewidth=1, alpha=0.7)
            if thresh[1] is not None: ax.axhline(thresh[1], color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(peak_t, color="#D32F2F", linestyle=":", linewidth=2, alpha=0.8)

    axes[-1].set_xlabel("Temps", fontsize=10)
    if np.issubdtype(type(time_axis[0]), np.datetime64):
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    filename = f"voie_{voie}_anomalie_{anomalie_id}_timestamp_{timestamp_str}.png"
    pic_folder = os.path.join(save_dir, f"anomalie_{anomalie_id}")
    os.makedirs(pic_folder, exist_ok=True)
    fig.savefig(os.path.join(pic_folder, filename), dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_rupture_anomaly(df_pandas, voie, seg_idx, breakpoints, save_dir):
    col_eff = f"efficace{voie}"
    signal = df_pandas[col_eff].values
    time_axis = df_pandas["time"].values

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f"Ruptures détectées — Voie {voie}", fontsize=14, fontweight="bold")
    ax.plot(time_axis, signal, color="#2196F3", linewidth=0.7, label="efficace")

    all_bkps = [0] + breakpoints + [len(signal)]
    cmap = plt.cm.get_cmap("Set3", len(all_bkps))
    for i in range(len(all_bkps) - 1):
        ax.axvspan(time_axis[all_bkps[i]], time_axis[min(all_bkps[i + 1], len(time_axis) - 1)], alpha=0.15, color=cmap(i))

    for bp in breakpoints:
        if bp < len(time_axis): ax.axvline(time_axis[bp], color="#D32F2F", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Temps", fontsize=10)
    ax.set_ylabel("Moyenne efficace", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if np.issubdtype(type(time_axis[0]), np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"voie{voie}_segment{seg_idx}_rupture.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_centered_rupture(df_pandas, voie, bp_idx, rupture_id, save_dir):
    start_idx = max(0, bp_idx - 10000)
    end_idx = min(len(df_pandas), bp_idx + 10000)
    window_df = df_pandas.iloc[start_idx:end_idx].copy()
    
    bp_time = df_pandas.iloc[bp_idx]["time"]
    display_time = bp_time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str = bp_time.strftime("%Y-%m-%d_%H-%M-%S")
        
    col_eff = f"efficace{voie}"
    time_axis = window_df["time"].values
    signal = window_df[col_eff].values

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Rupture {rupture_id} - Centrée (+/- 10000 pts) - Voie {voie}", fontsize=14, fontweight="bold")
    ax.plot(time_axis, signal, color="#2196F3", linewidth=1.5, label="efficace")
    ax.axvline(bp_time, color="#D32F2F", linestyle="--", linewidth=2, label="Rupture")
    ax.set_xlabel("Temps", fontsize=10)
    ax.set_ylabel("Moyenne efficace", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Heure rupture : {display_time}", fontsize=11)

    if np.issubdtype(type(time_axis[0]), np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    rup_folder = os.path.join(save_dir, f"rupture_{rupture_id}_zoom")
    os.makedirs(rup_folder, exist_ok=True)
    fig.savefig(os.path.join(rup_folder, f"voie{voie}_rupture_{rupture_id}_timestamp_{timestamp_str}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_oscillo_window(df_window, voie, original_anom_path, save_txt=True):
    if isinstance(df_window, cudf.DataFrame):
        df_window = df_window.to_pandas()

    fig, ax = plt.subplots(figsize=(12, 4))
    col_name = f"voie{voie}"
    time_axis = df_window["time"]
    values = df_window[col_name]
    
    ax.plot(time_axis, values, color="#D32F2F", linewidth=1.2, label=f"Voie {voie} (Oscillo HR)")
    ax.set_ylabel("Tension (V)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Analyse Oscilloscope - Zoom 1s (Voie {voie})", fontsize=12, fontweight="bold")
    
    if np.issubdtype(type(time_axis.iloc[0]), np.datetime64):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    
    plt.tight_layout()
    dir_name = os.path.dirname(original_anom_path)
    base_name = os.path.basename(original_anom_path)
    name_no_ext = os.path.splitext(base_name)[0]
    
    fig.savefig(os.path.join(dir_name, f"{name_no_ext}_oscillo.png"), dpi=150, bbox_inches="tight")
    
    if save_txt:
        txt_path = os.path.join(dir_name, f"{name_no_ext}_voie{voie}.txt")
        with open(txt_path, "w") as f:
            f.write("time\tvalue\n")
            for t, v in zip(time_axis, values): f.write(f"{t}\t{v}\n")
    plt.close(fig)

# ==============================================================================
# 5. EXECUTION HYBRIDE DES PIPELINES
# ==============================================================================
def run_pipeline_hybrid(df_gpu, output_dirs):
    summary = {v: {"pics": 0, "ruptures": 0} for v in range(1, NUM_VOIES + 1)}

    # ==========================================================================
    # BLOC 1 : Traitement des Pics (GPU)
    # ==========================================================================
    logger.info("--- BLOC 1 : Traitement des Pics (GPU : Ultra-rapide) ---")
    segments_peaks = segment_signal(df_gpu, window=WINDOW_SIZE_PEAK, overlap=0)
    for seg_idx, segment in segments_peaks:
        for voie in range(1, NUM_VOIES + 1):
            peak_idx_local = detect_peaks(segment, voie)
            if peak_idx_local is not None:
                summary[voie]["pics"] += 1
                plot_peak_anomaly(segment, voie, peak_idx_local, summary[voie]["pics"], output_dirs[voie]["pics"])

    # ==========================================================================
    # BLOC 2 : Traitement des Ruptures (CPU avec protection RAM)
    # ==========================================================================
    logger.info("--- BLOC 2 : Traitement des Ruptures (CPU avec protection RAM via Chunking) ---")
    
    # 1. On convertit en Pandas (on libère la mémoire GPU si besoin, mais ici c'est la RAM CPU le souci)
    df_cpu = df_gpu.to_pandas()
    
    # PARAMÈTRE DE SÉCURITÉ : On découpe le signal en morceaux pour ne pas saturer la RAM
    # 250 000 points par morceau est un excellent compromis pour l'algo Binseg/KMeans
    chunk_size = 250000 
    total_rows = len(df_cpu)
    
    for voie in range(1, NUM_VOIES + 1):
        all_exact_breaks = []
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            # Si le dernier morceau est trop petit (ex: < 10000 pts), on l'ignore ou l'associe
            if (end_idx - start_idx) < 10000 and start_idx > 0:
                continue
                
            logger.info(f"  -> Analyse Ruptures Voie {voie} : points {start_idx} à {end_idx}...")
            df_chunk = df_cpu.iloc[start_idx:end_idx].copy()
            
            # Appel de la détection sur le morceau
            chunk_breaks = detect_ruptures_cpu(df_chunk, voie)
            
            # On réajuste les index locaux du morceau pour retrouver les index globaux du fichier complet
            for bp in chunk_breaks:
                global_bp = start_idx + bp
                all_exact_breaks.append(global_bp)
                
            # Nettoyage manuel de la mémoire du chunk pour le garbage collector de Python
            del df_chunk
            
        # Suppression des doublons potentiels aux frontières
        all_exact_breaks = sorted(list(set(all_exact_breaks)))
        
        if all_exact_breaks:
            logger.info(f"  -> Génération des graphiques de ruptures pour la Voie {voie} ({len(all_exact_breaks)} trouvées)...")
            plot_rupture_anomaly(df_cpu, voie, "Complet", all_exact_breaks, output_dirs[voie]["ruptures"])
            for idx_rup, bp in enumerate(all_exact_breaks):
                plot_centered_rupture(df_cpu, voie, bp, idx_rup+1, output_dirs[voie]["ruptures"])
            summary[voie]["ruptures"] += len(all_exact_breaks)

    # Nettoyage final du gros DataFrame CPU
    del df_cpu
    return summary

def run_oscillo_block(base_output_dir=OUTPUT_DIR, oscillo_data_dir=OSCILLO_DIR):
    logger.info("--- BLOC 3 : Chargement et Corrélation Oscilloscope HR (GPU) ---")
    matches = find_oscillo_matches(base_output_dir, oscillo_data_dir)
    if not matches: return
        
    osc_cols = ["time", "voie1", "voie2", "voie3"]
    for osc_path, anomalies in matches.items():
        logger.info("Chargement GPU du fichier lourd : %s", os.path.basename(osc_path))
        try:
            # 1. Chargement des données brutes
            df_osc = cudf.read_csv(osc_path, header=None, names=osc_cols, sep=",")
            
            # --- CORRECTION DU BUG `to_datetime` POUR L'OSCILLO ---
            try:
                # Tentative directe sur le GPU
                df_osc["time"] = cudf.to_datetime(df_osc["time"])
            except Exception:
                # Secours : Conversion CPU de la colonne "time" uniquement pour esquiver l'erreur cuDF
                time_cpu = pd.to_datetime(df_osc["time"].to_pandas(), errors="coerce")
                df_osc["time"] = cudf.Series(time_cpu)
            # -----------------------------------------------------

            # Nettoyage et tri sur le GPU
            df_osc = df_osc.dropna(subset=["time"]).sort_values("time")
            
        except Exception as e:
            logger.error("Erreur chargement %s: %s", osc_path, e)
            continue
            
        for anom in anomalies:
            ts = anom["ts"]
            voie = anom["voie"]
            anom_path = anom["path"]
            
            start_window = ts - pd.Timedelta(seconds=1)
            end_window = ts + pd.Timedelta(seconds=1)
            
            # Filtre temporel ultra-rapide en VRAM
            df_window = df_osc[(df_osc["time"] >= start_window) & (df_osc["time"] <= end_window)]
            if df_window.empty: continue
                
            plot_oscillo_window(df_window, voie, anom_path)
# ==============================================================================
# 6. APPLICATION PRINCIPALE
# ==============================================================================
if __name__ == "__main__":
    logger.info("=== DÉMARRAGE DU PIPELINE HYBRIDE OPTIMISÉ ===")

    output_dirs = create_output_dirs(OUTPUT_DIR, NUM_VOIES)

    # Chargement initial sur GPU
    df_gpu = load_data(DATA_FILE)

    # Pipeline Intelligent (Pics sur GPU, Ruptures sur CPU)
    summary = run_pipeline_hybrid(df_gpu, output_dirs)

    # Zoom Oscilloscope sur GPU
    run_oscillo_block(base_output_dir=OUTPUT_DIR, oscillo_data_dir=OSCILLO_DIR)

    print("\n" + "="*50)
    print("BILAN DES ANOMALIES (ARCHITECTURES HYBRIDE TOURNÉE)")
    print("="*50)
    for voie, details in summary.items():
        print(f"Voie {voie}: {details['pics']} pics, {details['ruptures']} ruptures détectées.")
    print("="*50)