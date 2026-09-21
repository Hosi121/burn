#!/usr/bin/env python3
"""Compare complete ConvTranspose calls in separate, alternating processes."""

import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("variants", nargs="+", help="Probe executables in this directory")
parser.add_argument("--pairs", type=int, default=5)
parser.add_argument("--output", default="paired.json")
parser.add_argument("--cases", nargs="+")
parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
args = parser.parse_args()
root = Path(__file__).resolve().parent
rows = []
cases = args.cases or list(dict.fromkeys(
    row["case"] for row in json.loads((root.parent / "paired_summary.json").read_text())
))
for threads in args.threads:
    for case in cases:
        for pair in range(args.pairs):
            order = args.variants if pair % 2 == 0 else list(reversed(args.variants))
            for variant in order:
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                process = subprocess.run(
                    ["taskset", "-c", ",".join(map(str, range(threads))),
                     str(root / variant), case],
                    env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                    text=True, capture_output=True, check=True,
                )
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                rows.extend(
                    dict(json.loads(line), variant=variant, pair=pair, threads=threads,
                         process_user_s=after.ru_utime - before.ru_utime,
                         process_system_s=after.ru_stime - before.ru_stime,
                         process_minor_faults=after.ru_minflt - before.ru_minflt)
                    for line in process.stdout.splitlines()
                )
                (root / args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Completed: {threads} threads, {case}", flush=True)

summary = []
for threads in args.threads:
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
