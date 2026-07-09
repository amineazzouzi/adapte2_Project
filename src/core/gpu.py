"""
Détection GPU (CuPy) et préchauffage — source unique pour tout le pipeline
adapte2 (oscillo_analysis.py, to_data_lake.py). Le test est réel (pas juste
l'import) : il faut qu'un device CUDA réponde, sinon fallback NumPy.
"""

import numpy as np
import cupy as cp  # type: ignore

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


def setup_memory_pools():
    """Pool mémoire GPU + pinned memory (réduit l'overhead d'allocation ~30%)."""
    mem_pool    = cp.cuda.MemoryPool()
    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_allocator(mem_pool.malloc)
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
    return mem_pool, pinned_pool


def warmup_fft(nfft_size=2 * 10000 - 1):
    """
    Préchauffe un plan cuFFT pour une taille de fenêtre typique (élimine la
    latence de planification au 1er appel réel). nfft_size est une
    estimation — la taille réelle des fenêtres dépend des données (le
    découpage ne tronque plus à une taille fixe, cf. signal_processing.windowing).
    """
    d = cp.zeros(nfft_size, dtype=cp.float32)
    cp.fft.rfft(d, n=nfft_size)
    cp.cuda.Stream.null.synchronize()
    del d


def warmup_argsort(size=1_000_000):
    """Préchauffe cupy.argsort (tri/dédoublonnage de l'ETL — profil Phase 4)."""
    d = cp.random.rand(size, dtype=cp.float32)
    cp.argsort(d)
    cp.cuda.Stream.null.synchronize()
    del d
