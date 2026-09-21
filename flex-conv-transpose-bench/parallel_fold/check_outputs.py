#!/usr/bin/env python3
"""Compare complete output streams with the recorded base hashes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

root = Path(__file__).resolve().parent
expected = json.loads((root.parent / "output_final.json").read_text())
pipe = root / "output.pipe"
os.mkfifo(pipe)
results = []
try:
    for reference in expected:
        actual = {}

        def read_output():
            digest = hashlib.sha256()
            size = 0
            with pipe.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
            actual.update(sha256=digest.hexdigest(), bytes=size)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        subprocess.run(
            ["taskset", "-c", "0-3", str(root / "preserved"),
             reference["case"], str(pipe)],
            env={**os.environ, "RAYON_NUM_THREADS": "4"},
            stdout=subprocess.DEVNULL, check=True,
        )
        reader.join(timeout=30)
        equal = reference["baseline"] == actual
        results.append(dict(case=reference["case"], baseline=reference["baseline"],
                            candidate=actual, equal=equal))
        (root / "outputs.json").write_text(json.dumps(results, indent=2) + "\n")
        print(reference["case"], equal, flush=True)
        assert equal
finally:
    pipe.unlink()
