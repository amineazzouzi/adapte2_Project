"""
ETL : ingestion des fichiers .txt bruts (oscillo + TRMS) dans le data lake
Parquet partitionné. Voir to_data_lake.py pour le point d'entrée CLI.

Utilise un étage HDF5 intermédiaire pour trier/fusionner/dédupliquer sans
tout charger en RAM (les fichiers sources peuvent être volumineux).
"""

import gc
import re
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
import psutil

from src.core.gpu import GPU_AVAILABLE, cp, setup_memory_pools, warmup_argsort

OSCILLO_COLS = ["timestamp", "signal"]
TRMS_METRICS = ["min", "max", "moy", "efficace"]

# 'mixed' gère à la fois "%Y-%m-%d %H:%M:%S.%f" (oscillo) et "%Y-%m-%d %H:%M:%S" (TRMS post-traitement)
TS_FORMAT = "mixed"


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                       HELPERS MÉMOIRE / GPU                           ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def adaptive_chunk_size(base: int) -> int:
    """Ajuste le nombre de lignes par chunk selon la RAM libre (≥4 GB de marge)."""
    ram = free_ram_gb()
    if ram > 10: return base * 4
    if ram > 6:  return base * 2
    if ram > 3:  return base
    if ram > 1.5: return base // 2
    return base // 4


def _gpu_argsort(arr: np.ndarray, gpu_sort_threshold: int, mem_pool=None) -> np.ndarray:
    """argsort stable GPU pour les grands tableaux, CPU sinon."""
    if GPU_AVAILABLE and len(arr) >= gpu_sort_threshold:
        try:
            idx = cp.argsort(cp.asarray(arr), kind='stable').get()
            if mem_pool is not None:
                mem_pool.free_all_blocks()
            return idx
        except Exception:
            pass
    return np.argsort(arr, kind='stable')


def _gpu_dedup_mask(sorted_ts: np.ndarray, gpu_sort_threshold: int, mem_pool=None) -> np.ndarray:
    """Masque de déduplication (True là où le timestamp change) sur GPU ou CPU."""
    if GPU_AVAILABLE and len(sorted_ts) >= gpu_sort_threshold:
        try:
            ts_gpu   = cp.asarray(sorted_ts)
            mask     = cp.empty(len(ts_gpu), dtype=cp.bool_)
            mask[0]  = True
            mask[1:] = ts_gpu[1:] != ts_gpu[:-1]
            result   = mask.get()
            del ts_gpu, mask
            if mem_pool is not None:
                mem_pool.free_all_blocks()
            return result
        except Exception:
            pass
    return np.concatenate(([True], sorted_ts[1:] != sorted_ts[:-1]))


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║               h5py STREAMING  — LECTURE UNIQUE DU CSV                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _h5_fancy_read(h5f, key, idx):
    """Lecture h5py avec indices quelconques (tri proxy pour compatibilité)."""
    local_order = np.argsort(idx)
    h5_idx      = idx[local_order]
    restore     = np.argsort(local_order)
    return h5f[key][h5_idx][restore]


def oscillo_csv_to_h5(csv_path, h5_paths, chunk_size):
    """
    Lit le CSV oscillo UNE SEULE FOIS (engine='c') et remplit 3 HDF5 simultanément.
    Format CSV : timestamp,voie1,voie2,voie3  (séparateur virgule, pas de header)
    → 3× moins d'I/O disque vs. relire le fichier par voie.
    """
    handles = [h5py.File(str(p), 'w') for p in h5_paths]
    ts_ds   = [None] * 3
    sig_ds  = [None] * 3

    reader = pd.read_csv(csv_path, sep=',', engine='c', header=None,
                         dtype={0: str, 1: 'float32', 2: 'float32', 3: 'float32'},
                         chunksize=chunk_size)
    try:
        for chunk in reader:
            ts = pd.to_datetime(chunk[0], format=TS_FORMAT).astype(np.int64).values
            n  = len(ts)
            for i, col in enumerate([1, 2, 3]):
                vals = chunk[col].values.astype(np.float32)
                h5f  = handles[i]
                if ts_ds[i] is None:
                    kw = dict(maxshape=(None,), chunks=(chunk_size,), compression='lzf')
                    ts_ds[i]  = h5f.create_dataset('timestamp', data=ts,   **kw)
                    sig_ds[i] = h5f.create_dataset('signal',    data=vals, **kw)
                else:
                    old = ts_ds[i].shape[0]
                    ts_ds[i].resize(old + n, axis=0);  ts_ds[i][old:]  = ts
                    sig_ds[i].resize(old + n, axis=0); sig_ds[i][old:] = vals
    finally:
        for h5f in handles:
            h5f.close()


def trms_csv_to_h5(csv_path, h5_paths, chunk_size):
    """
    Lit le CSV TRMS UNE SEULE FOIS et remplit 3 HDF5 (une par voie).
    Format CSV : timestamp, [min,max,moy,eff]×3  (cols 0-12, sep virgule)
    """
    handles = [h5py.File(str(p), 'w') for p in h5_paths]
    ts_ds   = [None] * 3
    val_ds  = [{} for _ in range(3)]

    reader = pd.read_csv(csv_path, sep=',', engine='c', header=None,
                         chunksize=chunk_size)
    try:
        for chunk in reader:
            ts = pd.to_datetime(chunk[0], format=TS_FORMAT).astype(np.int64).values
            n  = len(ts)
            for i in range(3):
                start = 1 + i * 4
                src   = list(range(start, start + 4))
                h5f   = handles[i]
                if ts_ds[i] is None:
                    kw = dict(maxshape=(None,), chunks=(chunk_size,), compression='lzf')
                    ts_ds[i] = h5f.create_dataset('timestamp', data=ts, **kw)
                    for metric, sc in zip(TRMS_METRICS, src):
                        val_ds[i][metric] = h5f.create_dataset(
                            metric, data=chunk[sc].values.astype(np.float32), **kw)
                else:
                    old = ts_ds[i].shape[0]
                    ts_ds[i].resize(old + n, axis=0); ts_ds[i][old:] = ts
                    for metric, sc in zip(TRMS_METRICS, src):
                        val_ds[i][metric].resize(old + n, axis=0)
                        val_ds[i][metric][old:] = chunk[sc].values.astype(np.float32)
    finally:
        for h5f in handles:
            h5f.close()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                   H5 → PARQUET TRIÉ + FUSION                          ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def _parquets_to_h5(h5_path, pq_files, data_cols, chunk_size):
    """Fusionne plusieurs Parquet en un HDF5 sans concat en RAM."""
    with h5py.File(str(h5_path), 'w') as h5f:
        ts_ds  = None
        val_ds = {}
        for pq_file in pq_files:
            for batch in pq.ParquetFile(str(pq_file)).iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()
                ts = df['timestamp'].astype(np.int64).values
                n  = len(ts)
                if ts_ds is None:
                    kw = dict(maxshape=(None,), chunks=(chunk_size,), compression='lzf')
                    ts_ds = h5f.create_dataset('timestamp', data=ts, **kw)
                    for col in data_cols:
                        val_ds[col] = h5f.create_dataset(
                            col, data=df[col].values.astype(np.float32), **kw)
                else:
                    old = ts_ds.shape[0]
                    ts_ds.resize(old + n, axis=0); ts_ds[old:] = ts
                    for col in data_cols:
                        val_ds[col].resize(old + n, axis=0)
                        val_ds[col][old:] = df[col].values.astype(np.float32)


def h5_to_parquet_and_merge(h5_stage, target_pq, col_names, chunk_size, gpu_sort_threshold):
    """
    H5 staging → Parquet trié (GPU argsort) puis fusion/dédup avec l'existant.
    Appelé après oscillo_csv_to_h5 / trms_csv_to_h5, voie par voie.
    """
    mem_pool = None
    if GPU_AVAILABLE:
        mem_pool = cp.get_default_memory_pool()

    pq_new    = h5_stage.parent / "_new.parquet"
    h5_merge  = h5_stage.parent / "_merge.h5"
    pq_merged = h5_stage.parent / "_merged.parquet"
    data_cols = col_names[1:]

    schema = pa.schema(
        [('timestamp', pa.timestamp('ns'))] +
        [(c, pa.float32()) for c in data_cols]
    )

    try:
        # ── Étape 1 : H5 → Parquet trié (GPU argsort) ───────────────────
        with h5py.File(str(h5_stage), 'r') as h5f:
            total    = h5f['timestamp'].shape[0]
            sort_idx = _gpu_argsort(h5f['timestamp'][:], gpu_sort_threshold, mem_pool)
            with pq.ParquetWriter(str(pq_new), schema, compression='snappy') as writer:
                for start in range(0, total, chunk_size):
                    idx    = sort_idx[start: start + chunk_size]
                    arrays = {'timestamp': pa.array(_h5_fancy_read(h5f, 'timestamp', idx),
                                                    type=pa.timestamp('ns'))}
                    for col in data_cols:
                        arrays[col] = pa.array(_h5_fancy_read(h5f, col, idx), type=pa.float32())
                    writer.write_table(pa.table(arrays, schema=schema))
        h5_stage.unlink()

        if not target_pq.exists():
            pq_new.rename(target_pq)
            return

        # ── Étape 2 : Fusion via HDF5 staging (évite concat en RAM) ─────
        _parquets_to_h5(h5_merge, [target_pq, pq_new], data_cols, chunk_size)
        pq_new.unlink()

        with h5py.File(str(h5_merge), 'r') as h5f:
            all_ts = h5f['timestamp'][:]

        sort_idx  = _gpu_argsort(all_ts, gpu_sort_threshold, mem_pool)
        sorted_ts = all_ts[sort_idx]
        keep      = _gpu_dedup_mask(sorted_ts, gpu_sort_threshold, mem_pool)
        dedup_idx = sort_idx[keep]
        del all_ts, sort_idx, sorted_ts, keep
        gc.collect()

        with h5py.File(str(h5_merge), 'r') as h5f:
            total = len(dedup_idx)
            with pq.ParquetWriter(str(pq_merged), schema, compression='snappy') as writer:
                for start in range(0, total, chunk_size):
                    idx    = dedup_idx[start: start + chunk_size]
                    arrays = {'timestamp': pa.array(_h5_fancy_read(h5f, 'timestamp', idx),
                                                    type=pa.timestamp('ns'))}
                    for col in data_cols:
                        arrays[col] = pa.array(_h5_fancy_read(h5f, col, idx), type=pa.float32())
                    writer.write_table(pa.table(arrays, schema=schema))

        h5_merge.unlink()
        target_pq.unlink()
        pq_merged.rename(target_pq)

    finally:
        for tmp in [pq_new, h5_merge, pq_merged]:
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass
        gc.collect()


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                        FILE PROCESSORS                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def extract_date_from_path(path):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(path))
    return match.group(1) if match else None


def process_oscillo_file(file_path, boitier, date_str, data_lake_dir, chunk_size_base, gpu_sort_threshold):
    """
    1 lecture CSV → 3 HDF5 simultanés → 3 Parquet triés.
    engine='c' + format fixe = 5-10× plus rapide que engine='python' + inférence.
    """
    year, month, day = date_str.split("-")
    chunk_size  = adaptive_chunk_size(chunk_size_base)
    target_dirs = [
        Path(data_lake_dir) / boitier / "oscillo" / f"voie_{v}" /
        f"year={year}" / f"month={month}" / f"day={day}"
        for v in [1, 2, 3]
    ]
    for td in target_dirs:
        td.mkdir(parents=True, exist_ok=True)

    h5_stages = [td / "_stage.h5" for td in target_dirs]
    try:
        oscillo_csv_to_h5(file_path, h5_stages, chunk_size)
        for td, h5_stage in zip(target_dirs, h5_stages):
            h5_to_parquet_and_merge(h5_stage, td / "data.parquet", OSCILLO_COLS,
                                     chunk_size, gpu_sort_threshold)
            gc.collect()
    finally:
        for p in h5_stages:
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass


def process_trms_file(file_path, boitier, date_str, data_lake_dir, chunk_size_base, gpu_sort_threshold):
    """1 lecture CSV → 3 HDF5 simultanés → 3 Parquet triés (métriques TRMS)."""
    year, month, day = date_str.split("-")
    chunk_size  = adaptive_chunk_size(chunk_size_base)
    target_dirs = [
        Path(data_lake_dir) / boitier / "TRMS" / f"voie_{v}" /
        f"year={year}" / f"month={month}" / f"day={day}"
        for v in [1, 2, 3]
    ]
    for td in target_dirs:
        td.mkdir(parents=True, exist_ok=True)

    h5_stages = [td / "_stage.h5" for td in target_dirs]
    try:
        trms_csv_to_h5(file_path, h5_stages, chunk_size)
        col_names = ["timestamp"] + TRMS_METRICS
        for td, h5_stage in zip(target_dirs, h5_stages):
            h5_to_parquet_and_merge(h5_stage, td / "data.parquet", col_names,
                                     chunk_size, gpu_sort_threshold)
            gc.collect()
    finally:
        for p in h5_stages:
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║                   WORKER PARALLÈLE (par boitier)                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝

def process_boitier_group(file_tuples, data_lake_dir, chunk_size_base, gpu_sort_threshold):
    """
    Traite séquentiellement tous les fichiers d'UN boitier.
    Appelé dans un subprocess (spawn) → pas de conflit d'écriture entre boitiers.
    """
    results = []
    for f_str, date_str, boitier_id, is_trms in file_tuples:
        f_path = Path(f_str)
        t0 = time.time()
        try:
            if is_trms:
                process_trms_file(f_path, boitier_id, date_str, data_lake_dir,
                                   chunk_size_base, gpu_sort_threshold)
            else:
                process_oscillo_file(f_path, boitier_id, date_str, data_lake_dir,
                                      chunk_size_base, gpu_sort_threshold)
            results.append((f_path.name, boitier_id, None, time.time() - t0))
        except Exception as e:
            results.append((f_path.name, boitier_id, str(e), time.time() - t0))
    return results


class DataLakeWriter:
    """Orchestrateur de l'ETL : source .txt bruts -> data lake Parquet partitionné."""

    def __init__(self, config):
        self.config = config

    def run(self):
        import multiprocessing

        if GPU_AVAILABLE:
            setup_memory_pools()
            warmup_argsort()

        c = self.config
        t_global = time.time()
        print("=" * 75)
        print("ETL MULTI-BOITIERS & MULTI-DATES  —  h5py streaming + GPU argsort")
        gpu_label = "GPU: " + (str(cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
                                if GPU_AVAILABLE else "non disponible (CPU fallback)")
        print(f"RAM : {free_ram_gb():.1f} GB libre  |  {gpu_label}")
        print(f"Workers : {c.n_workers}  |  chunk base : {c.chunk_size_base:,} lignes  |  "
              f"seuil GPU sort : {c.gpu_sort_threshold:,}")
        print("=" * 75)

        all_txt_files = list(Path(c.source_root_dir).rglob("*.txt"))
        total_files   = len(all_txt_files)

        # Groupement par boitier — fichiers du même boitier traités séquentiellement
        # pour éviter les conflits d'écriture sur le même data.parquet.
        groups: dict = defaultdict(list)
        for f_path in all_txt_files:
            date_str      = extract_date_from_path(f_path) or time.strftime('%Y-%m-%d')
            boitier_match = re.search(r'(boitier_\d+)', str(f_path))
            boitier_id    = boitier_match.group(1) if boitier_match else "boitier_inconnu"
            is_trms       = "TRMS" in str(f_path) or "trms" in f_path.name
            groups[boitier_id].append((str(f_path), date_str, boitier_id, is_trms))

        n_groups = len(groups)
        print(f"\n  {total_files} fichiers → {n_groups} groupes (boitiers), "
              f"{c.n_workers} workers parallèles\n")

        # spawn : chaque worker importe le module proprement (CUDA-safe, pas de fork)
        mp_ctx    = multiprocessing.get_context('spawn')
        completed = 0

        with ProcessPoolExecutor(max_workers=c.n_workers, mp_context=mp_ctx) as executor:
            futures = {
                executor.submit(process_boitier_group, files, c.data_lake_dir,
                                 c.chunk_size_base, c.gpu_sort_threshold): boitier
                for boitier, files in groups.items()
            }
            for future in as_completed(futures):
                boitier = futures[future]
                try:
                    for name, bid, err, elapsed in future.result():
                        completed += 1
                        status = f"OK ({elapsed:.1f}s)" if err is None else f"ECHEC: {err}"
                        print(f"  [{completed:>5}/{total_files}] [{bid}] {name} ... {status}",
                              flush=True)
                except Exception as exc:
                    print(f"  [WORKER CRASH] [{boitier}] : {exc}")

        print("\n" + "=" * 75)
        print(f"DATA LAKE MIS A JOUR EN {time.time() - t_global:.1f}s")
        print(f"Destination : {c.data_lake_dir}")
        print("=" * 75)
