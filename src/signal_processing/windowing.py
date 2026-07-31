"""
Découpage et validation de fenêtres. Le découpage en fenêtres est PUREMENT
temporel (coupure sur écart > 0.5s) — aucune troncation à une taille fixe :
chaque fenêtre garde sa longueur naturelle. stack_variable_windows ne fait
que réconcilier des longueurs différentes en complétant (jamais en coupant),
après avoir retiré la composante continue de chaque fenêtre (voir
remove_dc_offset).
"""

import numpy as np
from scipy.ndimage import median_filter


def remove_dc_offset(all_windows):
    """
    Retire la composante continue (moyenne) de chaque fenêtre individuellement,
    pour qu'elles soient centrées sur zéro. Calculée sur le contenu réel de
    chaque fenêtre, AVANT le padding de stack_variable_windows (sinon la
    moyenne serait faussée par les zéros de complétion).
    """
    return [w - w.mean() for w in all_windows]


def stack_variable_windows(all_windows, all_time_arrays):
    """
    Empile des fenêtres de longueurs potentiellement différentes en un seul
    tableau 2D, SANS jamais tronquer : si des longueurs diffèrent, les plus
    courtes sont complétées par des zéros jusqu'à la longueur max observée
    (le contenu réel de chaque fenêtre est toujours conservé en entier).
    """
    if not all_windows:
        return np.array(all_windows), np.array(all_time_arrays, dtype=object)

    all_windows = remove_dc_offset(all_windows)

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


def filter_anomaly_windows(windows, n_segments=10, threshold=20):
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


def filter_by_peak_threshold(windows, threshold=30):
    """
    Contrainte supplémentaire sur les pics de la fenêtre entière (et non plus
    l'amplitude locale par segment, voir filter_anomaly_windows) : anomalie
    seulement si le pic positif (max) ET le pic négatif (min, en valeur
    absolue) dépassent tous deux threshold. Filtre les fenêtres où un seul
    des deux côtés montre une variation significative — pertinent maintenant
    que les fenêtres sont centrées sur zéro (voir remove_dc_offset), où un
    vrai signal perturbé est attendu symétrique.

    Vectorisé sur tout le batch en une seule opération NumPy.

    Retourne : array int (N,) avec 0 ou 1.
    """
    max_val = windows.max(axis=1)
    min_val = windows.min(axis=1)
    return ((max_val >= threshold) & (np.abs(min_val) >= threshold)).astype(int)


def classify_windows_by_peak_count(windows, neighborhood_size=20, peak_factor=50,
                                    merge_gap=50, low_level_pct=0.80,
                                    low_level_threshold=25):
    """
    Classe chaque fenêtre par nombre de pics isolés (pic_1, pic_2, pic_3, …) —
    méthode indépendante de filter_anomaly_windows / filter_by_peak_threshold,
    calculée avant le filtrage passe-bas et le calcul de similarité NCC (le
    pic est un phénomène haute fréquence que le passe-bas effacerait).

    Un point est un pic isolé si sa valeur absolue dépasse peak_factor fois
    la médiane des valeurs absolues de son voisinage (fenêtre glissante de
    2*neighborhood_size+1 points ; médiane plutôt que moyenne pour ne pas
    être faussée par un pic voisin proche) ET dépasse low_level_threshold
    (évite qu'un voisinage quasi nul ne déclenche un faux pic sur du bruit
    de fond).

    Des points-pics consécutifs séparés de moins de merge_gap échantillons
    sont fusionnés en un seul pic (un pic physique s'étale souvent sur
    plusieurs échantillons).

    Une fenêtre n'est classée "pic_N" que si le reste du signal est resté
    bas : au moins low_level_pct (défaut 80%) des échantillons de la fenêtre
    doivent être sous low_level_threshold en valeur absolue — sinon la
    fenêtre est laissée non classée (label "").

    Retourne : (n_peaks: array int (N,), labels: array str (N,))
               labels[i] == "" si la fenêtre ne qualifie pas (niveau de fond
               trop élevé) ou si aucun pic n'est détecté.
    """
    N, L = windows.shape
    abs_signal = np.abs(windows)

    local_median = median_filter(abs_signal, size=(1, 2 * neighborhood_size + 1), mode='nearest')
    is_peak_point = (abs_signal > peak_factor * local_median) & (abs_signal > low_level_threshold)

    frac_below = np.mean(abs_signal <= low_level_threshold, axis=1)
    qualifies = frac_below >= low_level_pct

    n_peaks = np.zeros(N, dtype=int)
    for i in range(N):
        idx = np.flatnonzero(is_peak_point[i])
        if len(idx) == 0:
            continue
        gaps = np.diff(idx)
        n_peaks[i] = 1 + int(np.sum(gaps > merge_gap))

    labels = np.array([
        f"pic_{n_peaks[i]}" if (qualifies[i] and n_peaks[i] > 0) else ""
        for i in range(N)
    ], dtype=object)

    return n_peaks, labels
