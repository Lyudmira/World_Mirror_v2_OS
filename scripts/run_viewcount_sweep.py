#!/usr/bin/env python3
"""
Sweep input view count: start from N views, decrease by step, using every-other
image (indices 0,2,4,...) from a sorted image directory. Stops on first successful
run (exit code 0).

Comments in English; print messages in English for logs.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def list_images_sorted(d: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    out: list[Path] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image_dir",
        type=Path,
        default=Path("/data/users/mia/current/lingbot-map/example/university"),
    )
    ap.add_argument(
        "--hyworld_dir",
        type=Path,
        default=Path("/data/users/mia/current/HY-World-2.0"),
    )
    ap.add_argument("--start", type=int, default=150, help="First trial: use this many views (after striding).")
    ap.add_argument("--step", type=int, default=10, help="Decrease view count by this each trial.")
    ap.add_argument("--min", type=int, default=10, help="Stop sweep at this N if still failing (then exit 1).")
    ap.add_argument(
        "--micromamba_env",
        type=str,
        default="hyworld2",
    )
    ap.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra args passed to pipeline, e.g. --target_size 518",
    )
    args = ap.parse_args()

    all_files = list_images_sorted(args.image_dir)
    if not all_files:
        print(f"No images in {args.image_dir}", file=sys.stderr)
        return 1

    # Every other image: 0,2,4,... (one in, one skipped)
    strided = all_files[::2]
    print(
        f"[Info] total={len(all_files)} files, after every-other: {len(strided)} files",
        flush=True,
    )

    if args.start > len(strided):
        print(
            f"[Warn] start={args.start} > strided count {len(strided)}; capping to {len(strided)}",
            flush=True,
        )

    n = min(args.start, len(strided))
    step = max(1, args.step)
    min_n = max(1, args.min)

    while n >= min_n:
        subset = strided[:n]
        print(f"\n[Try] n_views={n} (subset of strided list, first {n} files)", flush=True)

        with tempfile.TemporaryDirectory(prefix="wm_sweep_") as tmp:
            tdir = Path(tmp)
            for p in subset:
                dst = tdir / p.name
                src = p.resolve()
                if dst.exists():
                    raise RuntimeError(f"duplicate basename in subset: {p.name}")
                os.symlink(src, dst)

            inner = [
                "python",
                "-m",
                "hyworld2.worldrecon.pipeline",
                "--input_path",
                str(tdir),
                "--no_interactive",
                "--enable_bf16",
            ]
            if args.extra_args.strip():
                inner.extend(shlex.split(args.extra_args))

            if shutil.which("micromamba"):
                full = ["micromamba", "run", "-n", args.micromamba_env, *inner]
            else:
                full = [sys.executable, *inner[1:]]  # python -> current interpreter

            env = os.environ.copy()
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            print(f"[Run] cwd={args.hyworld_dir}\n[Run] " + " ".join(full), flush=True)
            r = subprocess.run(
                full,
                cwd=str(args.hyworld_dir),
                env=env,
            )
            if r.returncode == 0:
                print(
                    f"\n[OK] Succeeded with n_views={n} (every-other selection, strided pool size {len(strided)}).",
                    flush=True,
                )
                return 0
            print(
                f"[Fail] exit_code={r.returncode} for n_views={n}; next n={n - step}",
                flush=True,
            )

        n -= step

    print(
        f"\n[Stop] Reached n < {min_n} without success. Last tried was above.",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
