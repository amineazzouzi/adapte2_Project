"""
Dispatch de plusieurs runs de scripts/oscillo_analysis.py en parallèle sur
un serveur multi-GPU : un process par job (boitier/voie/dates), un GPU par
process en round-robin (CUDA_VISIBLE_DEVICES). Chaque run reste un process
Python isolé — plus robuste que de faire jongler un seul process Python
entre plusieurs devices CuPy, et ne touche à aucun code du pipeline lui-même.

Piloté par interface.py (seul point d'entrée du projet) ; trms_analysis.py
n'est pas concerné par ce dispatcher.
"""

import os
import subprocess
import sys
import threading
import time

from src.core.paths import date_suffix_for


def detect_gpu_count():
    """Nombre de GPUs disponibles : CuPy si possible, sinon nvidia-smi, sinon 1."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            return n
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
        n = len([l for l in out.stdout.splitlines() if l.strip()])
        if n > 0:
            return n
    except Exception:
        pass
    return 1


def _job_label(job):
    return f"{job['boitier']}_v{job['voie']}_{date_suffix_for(job['dates'])}"


class OscilloBatchRunner:
    """
    Lance oscillo_analysis.py une fois par job de config.jobs, en parallèle,
    un GPU par process (round-robin sur detect_gpu_count() GPUs, ou
    config.max_parallel si fourni). stdout/stderr de chaque job va dans
    config.log_dir/{label}.log.
    """

    def __init__(self, config, project_dir):
        self.config = config
        self.project_dir = project_dir

    def run(self, on_line=None, on_job_done=None, stop_flag=None):
        """
        on_line(label, line)     : appelé pour chaque ligne de stdout/stderr
                                    d'un job (en plus de l'écriture dans son
                                    fichier log, toujours faite).
        on_job_done(label, ret)  : appelé quand un job se termine.
        stop_flag()              : si retourne True, arrête tous les jobs en
                                    cours et n'en lance plus de nouveaux
                                    (utilisé par le bouton « Arrêter » de
                                    l'interface graphique).
        Les trois sont optionnels : sans eux, comportement CLI identique.
        """
        c = self.config
        n_gpus = detect_gpu_count()
        max_parallel = c.max_parallel or n_gpus
        os.makedirs(c.log_dir, exist_ok=True)

        print("=" * 70)
        print("BATCH OSCILLO — dispatch multi-GPU")
        print(f"   Jobs          : {len(c.jobs)}")
        print(f"   GPUs détectés : {n_gpus}")
        print(f"   Parallélisme  : {max_parallel}")
        print(f"   Logs          : {os.path.abspath(c.log_dir)}")
        print("=" * 70 + "\n")

        pending = list(c.jobs)
        running = {}   # slot_id -> (Popen, label)
        results = []
        stopped = False

        python_exe = c.python_executable or sys.executable

        def launch(slot_id, job):
            gpu_id = slot_id % n_gpus
            label  = _job_label(job)
            log_fh = open(os.path.join(c.log_dir, f"{label}.log"), "w")

            argv = [
                python_exe, "-m", "scripts.oscillo_analysis",
                "--data-lake-path", c.data_lake_path,
                "--boitier", job["boitier"],
                "--voie", str(job["voie"]),
                "--dates", *job["dates"],
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            print(f"[GPU {gpu_id}] lancement : {label}")
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=env, cwd=self.project_dir, text=True, bufsize=1,
            )

            def _reader():
                for line in proc.stdout:
                    log_fh.write(line)
                    log_fh.flush()
                    if on_line:
                        on_line(label, line)
                log_fh.close()

            threading.Thread(target=_reader, daemon=True).start()
            running[slot_id] = (proc, label)

        while pending or running:
            if stop_flag and stop_flag():
                stopped = True
                for proc, _label in running.values():
                    proc.terminate()
                pending.clear()

            while pending and len(running) < max_parallel:
                slot_id = next(i for i in range(max_parallel) if i not in running)
                launch(slot_id, pending.pop(0))

            time.sleep(0.5)
            for slot_id in list(running):
                proc, label = running[slot_id]
                ret = proc.poll()
                if ret is not None:
                    status = "OK" if ret == 0 else f"ÉCHEC (code {ret})"
                    print(f"[GPU {slot_id % n_gpus}] terminé : {label} -> {status}")
                    results.append((label, ret))
                    if on_job_done:
                        on_job_done(label, ret)
                    del running[slot_id]

        n_ok = sum(1 for _, ret in results if ret == 0)
        print("\n" + "=" * 70)
        if stopped:
            print(f"ARRÊTÉ PAR L'UTILISATEUR — {n_ok}/{len(results)} job(s) OK avant arrêt")
        else:
            print(f"TERMINÉ — {n_ok}/{len(results)} job(s) OK")
        for label, ret in results:
            if ret != 0:
                print(f"  ÉCHEC : {label} (voir {c.log_dir}/{label}.log)")
        print("=" * 70)
        return results
