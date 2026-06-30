"""Détection GPU (CuPy) avec fallback NumPy transparent."""

import numpy as np

# FIX CRITIQUE : import cupy DANS le try/except pour éviter le crash si non installé
try:
    import cupy as cp  # type: ignore
    cp.cuda.Device(0).compute_capability
    GPU_AVAILABLE = True
    _mem_pool = cp.cuda.MemoryPool()
    _pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_allocator(_mem_pool.malloc)
    cp.cuda.set_pinned_memory_allocator(_pinned_pool.malloc)
    # Préchauffage cuFFT : élimine la latence de planification au premier appel réel
    _d = cp.zeros(9999, dtype=cp.float32)
    cp.fft.rfft(_d)
    cp.cuda.Stream.null.synchronize()
    del _d
except Exception as _e:
    cp = None  # type: ignore
    GPU_AVAILABLE = False

xp = cp if GPU_AVAILABLE else np


def free_gpu_memory() -> None:
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()


def gpu_info() -> str:
    if GPU_AVAILABLE:
        return cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    return "CPU (NumPy fallback)"
