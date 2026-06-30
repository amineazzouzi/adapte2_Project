#!/usr/bin/env python3
# coding: utf-8
"""Point d'entrée CLI du pipeline Adapte2."""

import argparse
import json
import sys
from pathlib import Path

# Ajoute src/ au chemin Python pour permettre `from adapte2.xxx import yyy`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def cmd_analyze(args: argparse.Namespace) -> None:
    from adapte2.core.config import PipelineConfig, SignalConfig, GpuConfig, PathConfig
    from adapte2.analysis.pipeline import OscilloPipeline
    from adapte2.analysis.type_library import TypeLibrary
    from adapte2.reporting.dashboard import generate_dashboard
    from adapte2.core.gpu import gpu_info

    cfg = PipelineConfig(
        signal=SignalConfig(
            ncc_threshold=args.ncc_threshold,
            ncc_type_threshold=args.ncc_type_threshold,
            amplitude_threshold=args.amplitude_threshold,
            ncc_max_lag=args.ncc_max_lag,
            n_segments=args.n_segments,
        ),
        gpu=GpuConfig(batch_size=args.gpu_batch_size),
        paths=PathConfig(data_lake=args.data_lake),
        num_workers=args.workers,
    )

    type_lib = None
    if not args.no_type_library:
        lib_dir = Path(args.output).parent / cfg.paths.type_library
        type_lib = TypeLibrary.load(lib_dir)

    print(f"\n{'='*68}")
    print("PIPELINE ADAPTE2 — Analyse oscilloscope")
    print(f"  Signal  : {args.boitier} / voie_{args.voie} / {args.date}")
    print(f"  Device  : {gpu_info()}")
    print(f"  Sortie  : {Path(args.output).resolve()}")
    print(f"{'='*68}\n")

    pipeline = OscilloPipeline(config=cfg, type_library=type_lib)

    if args.files:
        signal_id = Path(args.files[0]).parent.name
        profile = pipeline.run_from_files(
            file_paths=args.files,
            output_dir=args.output,
            signal_id=signal_id,
        )
    else:
        profile = pipeline.run_from_datalake(
            boitier=args.boitier,
            voie=args.voie,
            date=args.date,
            output_dir=args.output,
        )

    if profile:
        generate_dashboard(profile, args.output)
        print(f"\n{'='*68}")
        print(f"OK PIPELINE TERMINE — {len(profile.get('events', []))} evenements")
        print(f"{'='*68}\n")


def cmd_correlate(args: argparse.Namespace) -> None:
    from adapte2.analysis.correlation import (
        compute_cooccurrence_matrix,
        compute_lead_lag,
        find_shared_types,
    )
    from adapte2.reporting.dashboard import generate_correlation_dashboard

    import json
    import pandas as pd

    profiles_with_dirs: list[tuple[dict, str]] = []
    for profile_dir in args.profiles:
        json_path = Path(profile_dir) / "signal_profile.json"
        if not json_path.exists():
            print(f"  ERREUR profil introuvable : {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for ev in raw["events"]:
            ev["ref_timestamp"] = pd.Timestamp(ev["ref_timestamp"])
            ev["window_timestamps"] = [pd.Timestamp(t) for t in ev.get("window_timestamps", [])]
        raw["t_start"] = pd.Timestamp(raw["t_start"]) if raw.get("t_start") else None
        raw["t_end"] = pd.Timestamp(raw["t_end"]) if raw.get("t_end") else None
        profiles_with_dirs.append((raw, str(profile_dir)))

    profiles = [p for p, _ in profiles_with_dirs]

    if len(profiles) < 2:
        print("Au moins 2 profils valides requis. Arret.")
        sys.exit(1)

    print(f"\n{'='*68}")
    print(f"PIPELINE CORRELATION — {len(profiles)} profil(s)")
    print(f"{'='*68}\n")

    cooccurrence_df = compute_cooccurrence_matrix(profiles, window_s=args.window_s)

    # Lead-lag pour chaque paire
    lead_lag_results: dict[str, dict] = {}
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            pa, pb = profiles[i], profiles[j]
            key = f"{pa['signal_id']} → {pb['signal_id']}"
            lead_lag_results[key] = compute_lead_lag(pa, pb, max_lag_s=60.0)

    shared_types = find_shared_types(profiles_with_dirs, ncc_threshold=args.ncc_threshold)

    generate_correlation_dashboard(
        profiles,
        cooccurrence_df,
        lead_lag_results,
        shared_types,
        args.output,
    )

    print(f"\n{'='*68}")
    print(f"OK CORRELATION TERMINEE")
    print(f"  Rapport : {Path(args.output) / 'rapport_correlation.html'}")
    print(f"{'='*68}\n")


def cmd_ui(args: argparse.Namespace) -> None:
    from adapte2.ui.app import App
    app = App()
    app.mainloop()


def cmd_etl(args: argparse.Namespace) -> None:
    """Lance l'ETL to_data_lake.py depuis le répertoire du projet."""
    import subprocess
    etl_script = Path(__file__).parent / "to_data_lake.py"
    if not etl_script.exists():
        print(f"ETL introuvable : {etl_script}")
        sys.exit(1)
    proc = subprocess.run([sys.executable, str(etl_script)], cwd=str(etl_script.parent))
    sys.exit(proc.returncode)


def cmd_calendar(args: argparse.Namespace) -> None:
    """Génère un calendrier mensuel depuis des dossiers de profils existants."""
    from adapte2.reporting.calendar_view import generate_calendar

    import json
    import pandas as pd

    events_by_date: dict[str, dict] = {}
    for profile_dir in args.profiles:
        json_path = Path(profile_dir) / "signal_profile.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for ev in raw["events"]:
            ev["ref_timestamp"] = pd.Timestamp(ev["ref_timestamp"])
            ev["window_timestamps"] = [pd.Timestamp(t) for t in ev.get("window_timestamps", [])]
        # Extraire la date depuis signal_id ou le nom du dossier
        date_str = raw.get("signal_id", "").split("/")[-1]
        if len(date_str) == 10:  # YYYY-MM-DD
            events_by_date[date_str] = raw

    generate_calendar(events_by_date, output_dir=args.output, month=args.month)


# ── Parseur ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adapte2",
        description="Pipeline d'analyse de signaux oscilloscopes agricoles.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── analyze ──────────────────────────────────────────────────────────────
    p_analyze = subparsers.add_parser("analyze", help="Analyser un signal depuis le data lake")
    p_analyze.add_argument("--boitier", required=True, help="ex: boitier_1")
    p_analyze.add_argument("--voie", type=int, required=True, help="ex: 1")
    p_analyze.add_argument("--date", required=True, help="ex: 2026-03-17")
    p_analyze.add_argument(
        "--output", type=Path, default=None,
        help="Dossier de sortie (auto si absent)"
    )
    p_analyze.add_argument(
        "--files", nargs="+", type=Path,
        help="Fichiers .txt bruts (alternative au data lake)"
    )
    p_analyze.add_argument(
        "--data-lake",
        default="/home/azzouzi/Bureau/Projet IA/adapte2_Project/data_lake",
        help="Chemin du data lake Parquet"
    )
    p_analyze.add_argument("--ncc-threshold", type=float, default=0.30)
    p_analyze.add_argument("--ncc-type-threshold", type=float, default=0.30)
    p_analyze.add_argument("--ncc-max-lag", type=int, default=1000)
    p_analyze.add_argument("--amplitude-threshold", type=float, default=50.0)
    p_analyze.add_argument("--n-segments", type=int, default=10)
    p_analyze.add_argument("--gpu-batch-size", type=int, default=2048)
    p_analyze.add_argument("--workers", type=int, default=4)
    p_analyze.add_argument(
        "--no-type-library", action="store_true",
        help="Désactiver la bibliothèque persistante de types"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # Résolution automatique du dossier de sortie
    def _resolve_output(args: argparse.Namespace) -> None:
        if args.output is None:
            args.output = Path(f"outputs_{args.boitier}_v{args.voie}_{args.date}")

    # ── correlate ─────────────────────────────────────────────────────────────
    p_corr = subparsers.add_parser(
        "correlate", help="Correler plusieurs profils de signaux"
    )
    p_corr.add_argument(
        "--profiles", nargs="+", type=Path, required=True,
        help="Dossiers contenant signal_profile.json"
    )
    p_corr.add_argument("--output", type=Path, default=Path("outputs_correlation"))
    p_corr.add_argument("--window-s", type=float, default=30.0,
                         help="Fenetre de co-occurrence en secondes")
    p_corr.add_argument("--ncc-threshold", type=float, default=0.30)
    p_corr.set_defaults(func=cmd_correlate)

    # ── ui ────────────────────────────────────────────────────────────────────
    p_ui = subparsers.add_parser("ui", help="Lancer l'interface graphique")
    p_ui.set_defaults(func=cmd_ui)

    # ── etl ───────────────────────────────────────────────────────────────────
    p_etl = subparsers.add_parser("etl", help="Lancer l'ETL .txt -> data lake Parquet")
    p_etl.set_defaults(func=cmd_etl)

    # ── calendar ──────────────────────────────────────────────────────────────
    p_cal = subparsers.add_parser("calendar", help="Générer un calendrier mensuel HTML")
    p_cal.add_argument(
        "--profiles", nargs="+", type=Path, required=True,
        help="Dossiers contenant signal_profile.json"
    )
    p_cal.add_argument("--output", type=Path, default=Path("outputs_calendar"))
    p_cal.add_argument("--month", default=None, help="ex: 2026-03")
    p_cal.set_defaults(func=cmd_calendar)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    # Résolution auto du dossier de sortie pour `analyze`
    if args.command == "analyze":
        _resolve_output(args)

    args.func(args)


if __name__ == "__main__":
    main()
