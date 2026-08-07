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
from sklearn.cluster import KMeans


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


def classify_windows_by_peak_kmeans(windows, search_width=10, min_cluster_separation=12):
    """
    Classe chaque fenêtre par nombre de pics isolés (pic_1, pic_2, pic_3, …)
    par segmentation + KMeans — méthode indépendante de filter_anomaly_windows
    / filter_by_peak_threshold, calculée avant le filtrage passe-bas et le
    calcul de similarité NCC (le pic est un phénomène haute fréquence que le
    passe-bas effacerait). Voir pic_kmeans_explorer.ipynb pour l'exploration
    interactive de cette approche.

    Pour chaque fenêtre :
      1. Découpe en segments de search_width échantillons.
      2. Score de chaque segment = max(signal**4) sur le segment (toujours
         positif, amplifie fortement l'écart entre le fond et un pic).
      3. KMeans (2 clusters) sur les scores des segments -> le cluster "pic"
         est celui dont le centre est le plus élevé.

    Un segment est retenu comme vrai pic si :
      1. Son score dépasse (ou atteint) le centre du cluster "pic" lui-même
         — pas juste l'appartenance au cluster KMeans (qui inclut aussi les
         segments plus faibles que la moyenne de ce cluster). Le ">=" (et pas
         ">" strict) est nécessaire : un segment seul dans le cluster "pic" a
         un score exactement égal au centre de son propre cluster (moyenne
         d'un seul point), donc ">" l'exclurait toujours.
      2. La séparation entre les deux centres (centre_pic / centre_normal)
         dépasse min_cluster_separation, sinon on considère qu'il n'y a pas
         de vrai pic dans la fenêtre (aucun segment retenu).

    Les segments-pics adjacents sont fusionnés en un seul pic (un pic
    physique s'étale souvent sur plusieurs segments).

    Retourne : (n_peaks: array int (N,), labels: array str (N,))
               labels[i] == "" si aucun pic n'est détecté dans la fenêtre.
    """
    N, L = windows.shape
    n_seg = L // search_width
    trimmed = windows[:, :search_width * n_seg].reshape(N, n_seg, search_width)
    seg_scores = np.max(trimmed.astype(np.float64) ** 4, axis=2)  # (N, n_seg)

    n_peaks = np.zeros(N, dtype=int)
    labels = np.full(N, "", dtype=object)

    for i in range(N):
        scores = seg_scores[i]
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(scores.reshape(-1, 1))
        centers = km.cluster_centers_.flatten()
        pic_cluster = int(np.argmax(centers))
        other_cluster = 1 - pic_cluster

        separation = centers[pic_cluster] / max(centers[other_cluster], 1e-9)
        if separation < min_cluster_separation:
            continue

        is_true_pic_seg = scores >= centers[pic_cluster]
        true_segs = np.flatnonzero(is_true_pic_seg)
        if len(true_segs) == 0:
            continue

        gaps = np.diff(true_segs)
        n_peaks[i] = 1 + int(np.sum(gaps > 1))
        labels[i] = f"pic_{n_peaks[i]}"

    return n_peaks, labels


def compute_window_energy(windows, median_filter_size=5):
    """
    Énergie "brute" de chaque fenêtre : somme des valeurs absolues du signal
    après filtre médian (lisse le bruit haute fréquence avant de sommer) —
    PAS de carré ni de puissance 4, volontairement différent du score de
    classify_windows_by_peak_kmeans. Sert à distinguer, parmi les fenêtres
    déjà classées "pic", celles qui comptent aussi comme "pic + forme" (voir
    pic_kmeans_explorer.ipynb, seuil validé = 10000 avec filtre taille 5).

    Vectorisé sur tout le batch en une seule opération NumPy/SciPy.

    Retourne : array float (N,)
    """
    filtered = median_filter(windows, size=(1, median_filter_size), mode='nearest')
    return np.sum(np.abs(filtered), axis=1)
