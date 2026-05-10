"""
Parallel Slippi replay regenerator.

Spawns N workers. Each worker:
  - has its own xvfb display + Dolphin User dir + comm.json + output dir
  - launches Slippi Playback AppImage on one replay at a time
  - waits for [GAME_END_FRAME] in dolphin stdout, then for output to flush
  - moves output file to the destination dir (renamed to match source basename)
  - moves to next replay

Why xvfb + AppImage instead of nogui: the nogui front-end built from the
slippi-Ishiiruka fork hangs after BootCore in IS_PLAYBACK mode (Core::IsRunning
becomes true but no EXI traffic flows). The GUI Playback build works correctly
when given a virtual display via xvfb-run.

Required env vars (or pass via CLI flags):
    MELEE_ISO              path to Super Smash Bros. Melee NTSC 1.02 ISO
    SLIPPI_PLAYBACK_DIR    extracted AppImage dir containing AppRun
                           (i.e. the squashfs-root from `Slippi_Playback-x86_64.AppImage --appimage-extract`)

Usage:
    python regen_batch.py <src_dir> <out_dir> [--workers N] [--timeout SEC] [--limit M]
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = REPO_ROOT / "scripts" / "Dolphin.ini.template"


def make_worker_dir(root: Path, worker_id: int, melee_iso: Path, template_path: Path) -> dict:
    """Create per-worker User dir + output dir + comm path."""
    base = root / f"worker_{worker_id:02d}"
    user = base / "User"
    out = base / "out"
    (user / "Config").mkdir(parents=True, exist_ok=True)
    (user / "Slippi").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    template = template_path.read_text()
    ini = (template
           .replace("{{REGEN_DIR}}", str(out))
           .replace("{{MELEE_ISO}}", str(melee_iso)))
    (user / "Config" / "Dolphin.ini").write_text(ini)
    return {"base": base, "user": user, "out": out, "comm": base / "comm.json"}


def run_one(slp_in: Path, slp_out: Path, dirs: dict, timeout: float) -> tuple[bool, str]:
    """Process one replay. Returns (ok, message)."""
    # Clean output dir of any prior file
    for f in dirs["out"].glob("Game_*.slp"):
        f.unlink()
    # Write comm.json - "mirror" mode triggers Hard FFW (10x+ speed)
    # via SlippiPlayback.cpp line 766: isHardFFW = (mode == "mirror")
    dirs["comm"].write_text(json.dumps({
        "mode": "mirror",
        "replay": str(slp_in),
        "isRealTimeMode": False,
        "outputOverlayFiles": False,
        "commandId": str(time.time_ns()),
    }))

    cmd = [
        "xvfb-run", "-a",
        "--server-args=-screen 0 320x240x16",
        str(dirs["apprun"]),
        "-e", str(dirs["iso"]),
        "-i", str(dirs["comm"]),
        "-b", "-v", "Null",
        "-u", str(dirs["user"]),
        "--cout",  # required so [GAME_END_FRAME] appears in stdout
    ]
    logfile = dirs["base"] / "dolphin.log"
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=open(logfile, "wb"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # so we can kill the whole group (xvfb-run + dolphin)
        )
    except FileNotFoundError as e:
        return False, f"missing tool: {e}"

    # Completion detection:
    #   1. Wait for [GAME_END_FRAME] in stdout — playback truly finished
    #   2. After that, wait 3 polls (6s) for the regenerate writer to flush
    #   3. Hard fallback: if no [GAME_END_FRAME] but file size has been stable
    #      for 8 polls (16s) AND > 1MB, accept (degraded mode)
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_polls = 0
    saw_game_end = False
    end_seen_at = None
    while time.monotonic() < deadline:
        time.sleep(2.0)
        # Look for [GAME_END_FRAME] line
        if not saw_game_end:
            try:
                with open(logfile, "rb") as f:
                    if b"[GAME_END_FRAME]" in f.read():
                        saw_game_end = True
                        end_seen_at = time.monotonic()
            except FileNotFoundError:
                pass

        produced = sorted(dirs["out"].glob("Game_*.slp"))
        if produced:
            sz = produced[-1].stat().st_size
            if sz == last_size and sz > 100_000:
                stable_polls += 1
            else:
                stable_polls = 0
                last_size = sz

            # Accept if game ended AND we've waited 3 stable polls (6s flush)
            if saw_game_end and stable_polls >= 3:
                _kill(proc)
                produced[-1].rename(slp_out)
                return True, f"ok ({sz} B, game_end=True)"
            # Fallback: no game-end but file size stable for 16s
            if not saw_game_end and stable_polls >= 8:
                _kill(proc)
                produced[-1].rename(slp_out)
                return True, f"ok-fallback ({sz} B, no game_end signal)"

        if proc.poll() is not None and not produced:
            return False, f"dolphin exited early code={proc.returncode}"
    _kill(proc)
    return False, f"timeout after {timeout}s (last_size={last_size}, game_end={saw_game_end})"


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=3)
    except (ProcessLookupError, PermissionError):
        pass


def worker_main(worker_id, work_root, out_dir, timeout, melee_iso, apprun, template_path, queue, results):
    dirs = make_worker_dir(work_root, worker_id, melee_iso, template_path)
    dirs["iso"] = melee_iso
    dirs["apprun"] = apprun
    while True:
        try:
            slp_in = queue.get(timeout=2)
        except Exception:
            break
        if slp_in is None:
            break
        slp_in = Path(slp_in)
        slp_out = out_dir / slp_in.name
        if slp_out.exists():
            results.put((str(slp_in), True, "skipped (exists)"))
            continue
        t0 = time.monotonic()
        ok, msg = run_one(slp_in, slp_out, dirs, timeout)
        dt = time.monotonic() - t0
        results.put((str(slp_in), ok, f"{msg} [worker {worker_id}, {dt:.1f}s]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="dir of source .slp files")
    ap.add_argument("dst", type=Path, help="dir to write regenerated .slp files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per replay")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--work-root", type=Path, default=Path.cwd() / ".regen-workdir",
                    help="scratch dir for per-worker User dirs / comm files / logs")
    ap.add_argument("--iso", type=Path, default=os.environ.get("MELEE_ISO"),
                    help="path to Melee NTSC 1.02 ISO (or set $MELEE_ISO)")
    ap.add_argument("--playback-dir", type=Path, default=os.environ.get("SLIPPI_PLAYBACK_DIR"),
                    help="extracted Slippi Playback AppImage dir containing AppRun "
                         "(or set $SLIPPI_PLAYBACK_DIR)")
    ap.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                    help="Dolphin.ini template (default: ./scripts/Dolphin.ini.template)")
    args = ap.parse_args()

    if not args.iso or not args.iso.exists():
        sys.exit(f"missing --iso / $MELEE_ISO (got {args.iso}); needs Melee NTSC 1.02 ISO")
    apprun = args.playback_dir / "AppRun" if args.playback_dir else None
    if not apprun or not apprun.exists():
        sys.exit(f"missing --playback-dir / $SLIPPI_PLAYBACK_DIR; expected to contain AppRun "
                 f"(got {args.playback_dir}). Run scripts/setup_playback.sh to download/extract.")
    if not args.template.exists():
        sys.exit(f"missing template at {args.template}")

    files = sorted(args.src.glob("*.slp"))
    if args.limit:
        files = files[:args.limit]
    args.dst.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    print(f"[main] {len(files)} replays, {args.workers} workers, timeout={args.timeout}s")
    print(f"[main] src={args.src} dst={args.dst}")
    print(f"[main] iso={args.iso} apprun={apprun}")

    q: mp.Queue = mp.Queue()
    rq: mp.Queue = mp.Queue()
    for f in files: q.put(str(f))
    for _ in range(args.workers): q.put(None)

    procs = []
    for wid in range(args.workers):
        p = mp.Process(target=worker_main,
                       args=(wid, args.work_root, args.dst, args.timeout,
                             args.iso, apprun, args.template, q, rq))
        p.start(); procs.append(p)

    done = ok = fail = 0
    t0 = time.monotonic()
    target = len(files)
    while done < target:
        try:
            slp, was_ok, msg = rq.get(timeout=10)
        except Exception:
            if all(not p.is_alive() for p in procs): break
            continue
        done += 1
        ok += int(was_ok); fail += int(not was_ok)
        rate = done / (time.monotonic() - t0)
        eta_min = (target - done) / max(rate, 1e-6) / 60
        status = "OK  " if was_ok else "FAIL"
        print(f"[{done:>4}/{target}] {status} {Path(slp).name} -- {msg} -- ({rate:.2f}/s, ETA {eta_min:.1f} min)")
        sys.stdout.flush()

    for p in procs: p.join(timeout=10)
    print(f"[main] done: ok={ok} fail={fail} elapsed={(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
