"""Chargement des fenêtres oscillo depuis le data lake Parquet."""

import os

import numpy as np
import pandas as pd

from src.signal_processing.windowing import stack_variable_windows


def load_from_datalake(datalake_path, boitier, voie, dates):
    """
    Charge les données oscillo depuis le data lake Parquet pour une ou
    plusieurs dates. Le découpage en fenêtres est PUREMENT temporel : dès
    qu'un écart entre deux points consécutifs dépasse 0.5s, une nouvelle
    fenêtre commence. Chaque fenêtre garde sa longueur naturelle telle
    qu'enregistrée (pas de troncation à une taille fixe). Les jours sont
    chargés dans l'ordre et concaténés : le pipeline produit ainsi un seul
    profil/rapport couvrant tous les jours demandés, pas un par jour.
    """
    if isinstance(dates, str):
        dates = [dates]

    all_windows     = []
    all_time_arrays = []

    for date in dates:
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
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Détecter les coupures (gaps > 0.5s) → autant de fenêtres indépendantes
        diffs = df['timestamp'].diff().dt.total_seconds().fillna(0)
        jump_indices  = np.where(diffs > 0.5)[0]
        start_indices = np.concatenate(([0], jump_indices))
        end_indices   = np.concatenate((jump_indices, [len(df)]))

        for s, e in zip(start_indices, end_indices):
            seg_sig = df['signal'].iloc[s:e].values.astype(np.float64)
            seg_ts  = df['timestamp'].iloc[s:e].values  # datetime64[ns]

            if len(seg_sig) < 100:
                continue

            all_windows.append(seg_sig)
            all_time_arrays.append(seg_ts)

    print(f"--- {len(all_windows)} fenêtres chargées "
          f"({boitier} / voie_{voie} / {len(dates)} jour(s)) ---")
    return stack_variable_windows(all_windows, all_time_arrays)
