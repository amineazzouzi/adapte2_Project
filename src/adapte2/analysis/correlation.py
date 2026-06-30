"""Co-occurrence temporelle, lead-lag et types partagés entre signaux."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..signal.ncc import compute_ncc_single


def compute_cooccurrence_matrix(
    profiles: list[dict],
    window_s: float = 30.0,
) -> pd.DataFrame:
    """Pour chaque paire (A, B) : combien de fois une anomalie sur A est suivie sur B.

    Retourne DataFrame NxN (signal_ids en index/colonnes).
    """
    sig_ids = [p["signal_id"] for p in profiles]
    n = len(sig_ids)
    matrix = np.zeros((n, n), dtype=int)

    for i, prof_a in enumerate(profiles):
        ts_a = sorted(
            pd.Timestamp(t)
            for ev in prof_a["events"]
            for t in ev.get("window_timestamps", [ev["ref_timestamp"]])
        )
        for j, prof_b in enumerate(profiles):
            if i == j:
                continue
            ts_b = sorted(
                pd.Timestamp(t)
                for ev in prof_b["events"]
                for t in ev.get("window_timestamps", [ev["ref_timestamp"]])
            )
            count = 0
            b_ptr = 0
            for ta in ts_a:
                while b_ptr < len(ts_b) and ts_b[b_ptr] < ta:
                    b_ptr += 1
                for tb in ts_b[b_ptr:]:
                    delta = (tb - ta).total_seconds()
                    if delta > window_s:
                        break
                    if 0 <= delta <= window_s:
                        count += 1
            matrix[i, j] = count

    return pd.DataFrame(matrix, index=sig_ids, columns=sig_ids)


def compute_lead_lag(
    profile_a: dict,
    profile_b: dict,
    max_lag_s: float = 60.0,
    bin_s: float = 5.0,
) -> dict:
    """Distribution des décalages temporels entre événements de A et B.

    Retourne {lags, counts, dominant_lag_s}. Un lag positif signifie que B suit A.
    """
    ts_a = sorted(pd.Timestamp(ev["ref_timestamp"]) for ev in profile_a["events"])
    ts_b = sorted(pd.Timestamp(ev["ref_timestamp"]) for ev in profile_b["events"])

    bins = np.arange(-max_lag_s, max_lag_s + bin_s, bin_s)
    counts = np.zeros(len(bins) - 1, dtype=int)

    for ta in ts_a:
        for tb in ts_b:
            delta = (tb - ta).total_seconds()
            if -max_lag_s <= delta <= max_lag_s:
                idx = int((delta + max_lag_s) / bin_s)
                if 0 <= idx < len(counts):
                    counts[idx] += 1

    lags = (bins[:-1] + bins[1:]) / 2
    dominant_lag_s = float(lags[np.argmax(counts)]) if counts.sum() > 0 else 0.0

    return {"lags": lags.tolist(), "counts": counts.tolist(), "dominant_lag_s": dominant_lag_s}


def _sig_key(signal_id: str) -> str:
    """Extrait 'boitier_X/voie_Y' depuis 'boitier_X/voie_Y/YYYY-MM-DD'."""
    parts = signal_id.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else signal_id


def find_shared_types(
    profiles_with_dirs: list[tuple[dict, str]],
    ncc_threshold: float = 0.30,
) -> list[dict]:
    """Compare les fenêtres de référence de chaque type entre signaux distincts.

    Retourne liste triée par NCC décroissante.
    """
    import os

    unique_keys = list(dict.fromkeys(_sig_key(p["signal_id"]) for p, _ in profiles_with_dirs))
    if len(unique_keys) < 2:
        return []

    # Agréger par clé signal (boitier/voie) — plusieurs jours possibles
    summaries: dict[str, dict] = {}
    for key in unique_keys:
        cluster_freqs: dict[int, list[float]] = defaultdict(list)
        cluster_imgs: dict[int, str] = {}
        cluster_wins: dict[int, np.ndarray] = {}
        for profile, pdir in profiles_with_dirs:
            if _sig_key(profile["signal_id"]) != key:
                continue
            refs_dir = os.path.join(pdir, "refs")
            for ev in profile["events"]:
                cid = ev["cluster_id"]
                cluster_freqs[cid].append(ev["ref_dom_freq"])
                if cid not in cluster_imgs and os.path.isdir(refs_dir):
                    for fname in sorted(os.listdir(refs_dir)):
                        if f"_type{cid}_ref.png" in fname:
                            cluster_imgs[cid] = os.path.join(refs_dir, fname)
                            break
                if cid not in cluster_wins:
                    npy = os.path.join(refs_dir, f"type{cid}_window.npy")
                    if os.path.exists(npy):
                        cluster_wins[cid] = np.load(npy).astype(np.float64)
        summaries[key] = {
            cid: {
                "mean_freq": sum(freqs) / len(freqs),
                "count": len(freqs),
                "ref_img": cluster_imgs.get(cid),
                "window": cluster_wins.get(cid),
            }
            for cid, freqs in cluster_freqs.items()
        }

    shared: list[dict] = []
    seen: set[tuple] = set()
    for i in range(len(unique_keys)):
        for j in range(i + 1, len(unique_keys)):
            key_a, key_b = unique_keys[i], unique_keys[j]
            for cid_a, info_a in summaries[key_a].items():
                for cid_b, info_b in summaries[key_b].items():
                    wa = info_a.get("window")
                    wb = info_b.get("window")
                    if wa is None or wb is None:
                        continue
                    n = min(len(wa), len(wb))
                    ncc_val = compute_ncc_single(wa[:n], wb[:n])
                    if ncc_val < ncc_threshold:
                        continue
                    pair_key = (key_a, cid_a, key_b, cid_b)
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)
                    shared.append(
                        {
                            "ncc": ncc_val,
                            "sig_a": key_a, "cid_a": cid_a,
                            "freq_a": info_a["mean_freq"], "count_a": info_a["count"],
                            "img_a": info_a["ref_img"],
                            "sig_b": key_b, "cid_b": cid_b,
                            "freq_b": info_b["mean_freq"], "count_b": info_b["count"],
                            "img_b": info_b["ref_img"],
                        }
                    )

    shared.sort(key=lambda x: x["ncc"], reverse=True)
    print(f"  -> {len(shared)} type(s) partagé(s) (NCC >= {ncc_threshold:.2f})")
    return shared
