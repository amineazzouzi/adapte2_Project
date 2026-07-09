"""
Découpage et validation de fenêtres. Le découpage en fenêtres est PUREMENT
temporel (coupure sur écart > 0.5s) — aucune troncation à une taille fixe :
chaque fenêtre garde sa longueur naturelle. stack_variable_windows ne fait
que réconcilier des longueurs différentes en complétant (jamais en coupant).
"""

import numpy as np


def stack_variable_windows(all_windows, all_time_arrays):
    """
    Empile des fenêtres de longueurs potentiellement différentes en un seul
    tableau 2D, SANS jamais tronquer : si des longueurs diffèrent, les plus
    courtes sont complétées par des zéros jusqu'à la longueur max observée
    (le contenu réel de chaque fenêtre est toujours conservé en entier).
    """
    if not all_windows:
        return np.array(all_windows), np.array(all_time_arrays, dtype=object)

    lengths = {len(w) for w in all_windows}
    if len(lengths) > 1:
        max_len = max(lengths)
        print(f"  ⚠ longueurs de fenêtre différentes détectées {sorted(lengths)} "
              f"-> complétées à {max_len} pts (jamais tronquées)")
        for i, (w, t) in enumerate(zip(all_windows, all_time_arrays)):
            if len(w) < max_len:
                pad = max_len - len(w)
                all_windows[i] = np.pad(w, (0, pad), 'constant')
                if len(t) > 1:
                    dt = t[1] - t[0]
                    t = np.concatenate([t, [t[-1] + (k + 1) * dt for k in range(pad)]])
                else:
                    t = np.concatenate([t, [t[-1]] * pad])
                all_time_arrays[i] = t

    return np.array(all_windows), np.array(all_time_arrays, dtype=object)


def filter_anomaly_windows(windows, n_segments=10, threshold=40):
    """
    Classe chaque fenêtre comme anomalie ou non selon son amplitude locale.

    Pour chaque fenêtre :
      1. Divise le signal en n_segments segments égaux.
      2. Calcule (max - min) sur chaque segment.
      3. Calcule la moyenne de ces n_segments valeurs.
      4. Si cette moyenne > threshold → anomalie (1), sinon 0.

    Vectorisé sur tout le batch en une seule opération NumPy.
    Si la longueur du signal n'est pas divisible par n_segments, les
    derniers points excédentaires sont ignorés (troncature).

    Retourne : array int (N,) avec 0 ou 1.
    """
    N, L = windows.shape
    seg_len = L // n_segments
    # Troncature pour alignement → shape (N, n_segments, seg_len)
    trimmed = windows[:, :seg_len * n_segments].reshape(N, n_segments, seg_len)
    seg_ranges = trimmed.max(axis=2) - trimmed.min(axis=2)   # (N, n_segments)
    mean_range = seg_ranges.mean(axis=1)                      # (N,)
    return (mean_range > threshold).astype(int)
