"""Reset all saved training data, models, gestures, and practice history."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 5000


def targets(keep_history: bool) -> list[Path]:
    paths = [
        ROOT / "data" / "samples.npz",
        ROOT / "data" / "gestures.json",
        ROOT / "data" / "thumbs",
        ROOT / "models" / "model.joblib",
        ROOT / "models" / "classes.json",
    ]
    if not keep_history:
        paths.append(ROOT / "data" / "practice.json")
    return paths


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def free_port(port: int) -> list[str]:
    """Kill any process listening on `port`. Returns the PIDs killed."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("Skipping port reset: 'lsof' is not available on this system.")
        return []

    pids = [pid for pid in result.stdout.split() if pid]
    for pid in pids:
        subprocess.run(["kill", "-9", pid], check=False)
    return pids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete saved app data and model artifacts."
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="preserve saved practice history",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    parser.add_argument(
        "--port",
        type=int,
        nargs="?",
        const=DEFAULT_PORT,
        default=None,
        metavar="PORT",
        help=f"also kill whatever process is listening on PORT "
        f"(default {DEFAULT_PORT} if flag given with no value)",
    )
    args = parser.parse_args()

    keep_history = args.keep_history and not args.include_history
    paths = targets(keep_history)
    existing = [path for path in paths if path.exists()]
    if not existing and args.port is None:
        print("Nothing to reset.")
        return

    if existing:
        print("The following will be deleted:")
        for path in existing:
            print(f"  {path}")
    if args.port is not None:
        print(f"Will also free port {args.port} (kill any process using it).")

    if not args.yes:
        answer = input("Continue? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Reset cancelled.")
            return

    for path in existing:
        remove(path)

    if existing:
        print("Reset complete. Practice history was kept." if keep_history
            else "Reset complete, including practice history.")

    if args.port is not None:
        killed = free_port(args.port)
        if killed:
            print(f"Freed port {args.port} (killed PID(s): {', '.join(killed)}).")
        else:
            print(f"Port {args.port} was already free.")


if __name__ == "__main__":
    main()
