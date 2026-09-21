#!/usr/bin/env python3
"""Compare full output bytes without stored tensor files."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

root = Path(__file__).resolve().parent
cases = [(r["dtype"], r["case"]) for r in json.loads(
    (root / "production_summary.json").read_text()) if r["threads"] == 4]
cases += [(r["dtype"], r["case"]) for r in json.loads(
    (root / "production_types_summary.json").read_text()) if r["threads"] == 4]
pipe = root / "output.pipe"
os.mkfifo(pipe)
results = []
try:
    for dtype, case in cases:
        record = dict(dtype=dtype, case=case, threads=4)
        for variant in ["baseline", "candidate"]:
            digest_result = {}

            def read_output():
                digest = hashlib.sha256()
                size = 0
                with pipe.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                digest_result.update(sha256=digest.hexdigest(), bytes=size)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            subprocess.run(
                ["taskset", "-c", "0-3", str(root / variant), case, str(pipe)],
                env={**os.environ, "RAYON_NUM_THREADS": "4", "PROBE_DTYPE": dtype},
                stdout=subprocess.DEVNULL, check=True, timeout=120,
            )
            reader.join(timeout=30)
            assert not reader.is_alive() and digest_result["bytes"] > 0
            record[variant] = digest_result
        record["equal"] = record["baseline"] == record["candidate"]
        results.append(record)
        (root / "outputs.json").write_text(json.dumps(results, indent=2) + "\n")
        print(dtype, case, record["equal"], flush=True)
        assert record["equal"]
finally:
    pipe.unlink()
