"""Download the NASA SMAP/MSL telemetry into the configured data directory.

The original telemanom S3 link in NASA's README is dead (403), so this pulls
from a maintained parquet mirror on HuggingFace plus NASA's own
labeled_anomalies.csv from the telemanom repository.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import urllib.request

from config import DATA_DIR

BASE = "https://huggingface.co/datasets/appleparan/telemanom/resolve/main/"
API = "https://huggingface.co/api/datasets/appleparan/telemanom/tree/main/"


def api(path: str):
    return json.load(urllib.request.urlopen(API + path, timeout=60))


def build_jobs() -> list[tuple[str, os.PathLike]]:
    """(remote path, local destination) pairs, anchored to DATA_DIR.

    Paths are resolved against config.DATA_DIR rather than the working
    directory: the documented invocation is `python src/fetch_data.py` from the
    prototype root, and cwd-relative paths would drop the dataset somewhere
    nothing else looks for it.
    """
    jobs = []
    for split in ("train", "test"):
        (DATA_DIR / split).mkdir(parents=True, exist_ok=True)
        for entry in api(f"data/{split}"):
            if entry["path"].endswith(".parquet"):
                name = os.path.basename(entry["path"])
                jobs.append((entry["path"], DATA_DIR / split / name))
    jobs.append(("labeled_anomalies.csv", DATA_DIR / "labeled_anomalies.csv"))
    return jobs


def fetch(job) -> str:
    src, dst = job
    if dst.exists() and dst.stat().st_size > 0:
        return f"{dst.name} (cached)"
    urllib.request.urlretrieve(BASE + src, dst)
    return f"{dst.name} {dst.stat().st_size}B"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    with cf.ThreadPoolExecutor(8) as pool:
        for _ in pool.map(fetch, jobs):
            pass
    print(f"downloaded {len(jobs)} files into {DATA_DIR}")


if __name__ == "__main__":
    main()
