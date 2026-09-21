#!/usr/bin/env python3
"""Compare complete ConvTranspose calls in separate, alternating processes."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("variants", nargs="+", help="Probe executables in this directory")
parser.add_argument("--pairs", type=int, default=5)
parser.add_argument("--output", default="paired.json")
args = parser.parse_args()
root = Path(__file__).resolve().parent
rows = []
for threads in [1, 4]:
    for pair in range(args.pairs):
        order = args.variants if pair % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            process = subprocess.run(
                ["taskset", "-c", "0" if threads == 1 else "0-3", str(root / variant)],
                env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                text=True, capture_output=True, check=True,
            )
            rows.extend(
                dict(json.loads(line), variant=variant, pair=pair, threads=threads)
                for line in process.stdout.splitlines()
            )
            (root / args.output).write_text(json.dumps(rows, indent=2) + "\n")
            print(f"Completed: {threads} threads, pair {pair + 1}, {variant}", flush=True)

summary = []
for threads in [1, 4]:
    for case in dict.fromkeys(r["case"] for r in rows):
        entry = {"case": case, "threads": threads}
        for variant in args.variants:
            selected = [r for r in rows if r["case"] == case
                        and r["threads"] == threads and r["variant"] == variant]
            times = [r["ms"] for r in selected]
            entry[variant] = {
                "ms": statistics.median(times), "range_ms": [min(times), max(times)],
                "peak_bytes": max(r["peak_bytes"] for r in selected),
                "output_bytes": selected[0]["output_bytes"],
            }
        summary.append(entry)
(root / args.output.replace(".json", "_summary.json")).write_text(
    json.dumps(summary, indent=2) + "\n")
