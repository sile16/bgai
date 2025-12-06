#!/usr/bin/env python
"""CLI helper for generating and inspecting bearoff databases.

Wraps the Rust `bearoff` module:
- generate: create a packed upper-triangle table (uint24 fixed) and write a JSON header.
- show-header: print header for an existing table (reads companion JSON if present, otherwise infers basics).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from bearoff import generate_packed_bearoff


def solve_size_to_n(entries: int) -> int:
    """Infer n from packed upper-triangle length."""
    n = int((math.isqrt(1 + 8 * entries) - 1) // 2)
    if n * (n + 1) // 2 != entries:
        raise ValueError(f"Packed length {entries} does not correspond to a valid n")
    return n


def cmd_generate(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path}")

    header_json, arr = generate_packed_bearoff(
        max_checkers=args.max_checkers,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
    )
    header: Dict[str, Any] = json.loads(header_json)
    arr_np = np.array(arr, copy=False)

    np.save(out_path, arr_np)
    header_path = out_path.with_suffix(out_path.suffix + ".json")
    with header_path.open("w") as f:
        json.dump(header, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.1f} MB), shape={arr_np.shape}, dtype={arr_np.dtype}")
    print(f"header -> {header_path}")


def cmd_show_header(args: argparse.Namespace) -> None:
    path = Path(args.path)
    header_path = path.with_suffix(path.suffix + ".json")
    if header_path.exists():
        header = json.loads(header_path.read_text())
        print(json.dumps(header, indent=2))
        return

    # Fallback: infer basic metadata from the npy file
    arr = np.load(path, mmap_mode="r")
    entries = arr.shape[0]
    slots = arr.shape[1] if arr.ndim > 1 else 1
    n = solve_size_to_n(entries)
    dtype = str(arr.dtype)
    print(
        json.dumps(
            {
                "inferred": True,
                "entries": entries,
                "n_positions": n,
                "slots": slots,
                "dtype": dtype,
                "shape": list(arr.shape),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bearoff DB helper (generator + header viewer).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="Generate packed upper-triangle bearoff DB (uint24 fixed).")
    pg.add_argument("--max-checkers", type=int, required=True, help="Checkers per side (e.g., 10, 15).")
    pg.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy path (header will be written to <output>.json).",
    )
    pg.add_argument("--tolerance", type=float, default=0.0, help="API placeholder (unused).")
    pg.add_argument("--max-iter", type=int, default=0, help="API placeholder (unused).")
    pg.add_argument("--overwrite", action="store_true", help="Overwrite existing output file.")
    pg.set_defaults(func=cmd_generate)

    ph = sub.add_parser("show-header", help="Show header for an existing DB (.json if present, otherwise inferred).")
    ph.add_argument("path", type=str, help="Path to .npy file (and optional .json header).")
    ph.set_defaults(func=cmd_show_header)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
