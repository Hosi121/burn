#!/usr/bin/env python3
"""Run the required checks and keep their exit codes."""

import json
import os
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parent
repo = Path(os.environ.get("BURN_REPO", str(Path.cwd())))
checks = [
    ("run_checks", ["cargo", "run-checks"]),
    ("clippy_targets", ["cargo", "clippy", "-p", "burn-flex", "--all-targets", "--", "--deny", "warnings"]),
    ("bench_check", ["cargo", "bench", "-p", "burn-backend-tests", "--bench", "conv_transpose_ops",
                     "--no-default-features", "--features", "std,flex-simd,flex-rayon", "--", "--test"]),
    ("diff_check", ["git", "diff", "--check"]),
]
results = []
for name, command in checks:
    start = time.monotonic()
    with (root / (name + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "CARGO_BUILD_JOBS": "6", "CARGO_INCREMENTAL": "0", "BURN_DEVICE": "flex"})
    results.append(dict(name=name, command=command, exit_code=result.returncode,
                        elapsed_s=time.monotonic() - start))
    (root / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
