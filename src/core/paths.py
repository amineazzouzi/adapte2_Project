"""
Conventions de nommage des dossiers/identifiants de signal — source unique
utilisée par oscillo_analysis.py et oscillo_correlation.py, pour qu'un
profil produit par l'un soit toujours retrouvable par l'autre.
"""


def date_suffix_for(dates):
    """'2026-03-17' si un seul jour, '2026-03-17_a_2026-03-19' si plage."""
    d0, d1 = dates[0], dates[-1]
    return d0 if d0 == d1 else f"{d0}_a_{d1}"


def output_dir_for(boitier, voie):
    """
    Un seul dossier par signal (boitier+voie), quel que soit le nombre de
    jours combinés — pas un dossier par jour. Un nouveau run pour le même
    signal remplace le précédent.
    """
    return f"outputs_{boitier}_v{voie}"


def signal_id_for(boitier, voie, dates):
    """Identifiant affiché/sérialisé dans signal_profile.json (informatif)."""
    return f"{boitier}/voie_{voie}/{date_suffix_for(dates)}"


def sig_key_from_signal_id(signal_id):
    """Extrait 'boitier_X/voie_Y' depuis 'boitier_X/voie_Y/YYYY-MM-DD' (ou
    une plage de dates) — l'inverse de signal_id_for, sans la partie date.
    Utilisé par oscillo_correlation.py pour regrouper les événements par
    signal, indépendamment du jour."""
    parts = signal_id.split('/')
    return '/'.join(parts[:2]) if len(parts) >= 2 else signal_id


def profile_dir_for(project_dir, boitier, voie):
    """Retrouve le dossier de sortie d'oscillo_analysis.py pour un signal donné."""
    import os
    return os.path.join(project_dir, output_dir_for(boitier, voie))
