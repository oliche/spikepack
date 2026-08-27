"""Benchmark spikepack (Blosc directory shard vs Zarr) on the 11 standard IBL benchmark insertions.

Loads spike times + cluster labels for each PID directly from Alyx/ONE (no good-unit
filtering — the full recording), encodes at the frozen bwm_ephys default of 100 us,
writes both container formats, and reports compression ratio / round-trip fidelity /
timing for each. Idempotent: re-running skips PIDs whose results are already cached
in RESULTS_PATH.

Usage
-----
uv run python scripts/benchmark_pids.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spikepack import read_blosc, read_zarr, write_blosc, write_zarr  # noqa: E402

QUANTIZATION_US = 100
PIDS = [
    "1a276285-8b0e-4cc9-9f0a-a3a002978724",  # 00 - Benchmark PIDS start
    "1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
    "6638cfb3-3831-4fc2-9327-194b76cf22e1",
    "749cb2b7-e57e-4453-a794-f6230e4d0226",
    "d7ec0892-0a6c-4f4f-9d8f-72083692af5c",
    "da8dfec1-d265-44e8-84ce-6ae9c109b8bd",  # 05
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",  # 06 - amazing CSD
    "dc7e9403-19f7-409f-9240-05ee57cb7aea",  # static noise: positive spikes
    "e8f9fba4-d151-4b00-bee7-447f0f3e752c",
    "eebcaf65-7fa4-4118-869d-a084e84530e2",
    "fe380793-8035-414e-b000-09bfe5ece92a",  # Benchmark PIDS stop
]

ROOT_OUTPUT = Path.home().joinpath("scratch", "spikepack_benchmark")
RESULTS_PATH = Path(__file__).resolve().parents[1].joinpath("reports", "benchmark_pids_results.json")


def bench_one(pid: str, one) -> dict:
    from brainbox.io.one import SpikeSortingLoader

    ssl = SpikeSortingLoader(pid=pid, one=one)
    spikes = one.load_object(
        ssl.eid, "spikes", collection=f"alf/{ssl.pname}/pykilosort", attribute=["times", "clusters"]
    )
    times = np.asarray(spikes["times"], dtype=np.float64)
    clusters = np.asarray(spikes["clusters"], dtype=np.int32)
    n = int(times.size)
    raw_bytes = int(times.nbytes + clusters.nbytes)

    out_dir = ROOT_OUTPUT.joinpath(pid)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_meta = {"pid": str(pid), "eid": str(ssl.eid), "probe_name": str(ssl.pname)}

    t0 = time.perf_counter()
    blosc_bytes = write_blosc(
        out_dir.joinpath("blosc_shard"),
        times_seconds=times,
        labels=clusters,
        quantization_us=QUANTIZATION_US,
        extra_meta=extra_meta,
    )
    t_blosc_write = time.perf_counter() - t0
    t0 = time.perf_counter()
    blosc_out = read_blosc(out_dir.joinpath("blosc_shard"))
    t_blosc_read = time.perf_counter() - t0

    zarr_path = out_dir.joinpath("zarr_store.zarr")
    t0 = time.perf_counter()
    write_zarr(zarr_path, times_seconds=times, labels=clusters, quantization_us=QUANTIZATION_US, extra_meta=extra_meta)
    t_zarr_write = time.perf_counter() - t0
    zarr_bytes = int(sum(p.stat().st_size for p in zarr_path.rglob("*") if p.is_file()))
    t0 = time.perf_counter()
    zarr_out = read_zarr(zarr_path)
    t_zarr_read = time.perf_counter() - t0

    max_err_blosc = float(np.max(np.abs(blosc_out["times"] - times))) if n else 0.0
    max_err_zarr = float(np.max(np.abs(zarr_out["times"] - times))) if n else 0.0
    labels_ok = bool(np.array_equal(blosc_out["labels"], clusters)) and bool(
        np.array_equal(zarr_out["labels"], clusters)
    )

    return dict(
        pid=pid,
        eid=str(ssl.eid),
        probe_name=str(ssl.pname),
        n_spikes=n,
        n_units=int(np.unique(clusters).size),
        duration_s=float(times[-1] - times[0]) if n else 0.0,
        raw_bytes=raw_bytes,
        blosc_bytes=int(blosc_bytes),
        zarr_bytes=zarr_bytes,
        cr_blosc=raw_bytes / blosc_bytes,
        cr_zarr=raw_bytes / zarr_bytes,
        blosc_write_s=t_blosc_write,
        blosc_read_s=t_blosc_read,
        zarr_write_s=t_zarr_write,
        zarr_read_s=t_zarr_read,
        max_abs_err_blosc_s=max_err_blosc,
        max_abs_err_zarr_s=max_err_zarr,
        labels_round_trip_ok=labels_ok,
    )


def main() -> None:
    from one.api import ONE

    results: dict[str, dict] = {}
    if RESULTS_PATH.exists():
        results = {r["pid"]: r for r in json.loads(RESULTS_PATH.read_text())}

    one = ONE()
    for pid in PIDS:
        if pid in results:
            print(f"{pid[:8]}: cached, skipping")
            continue
        print(f"{pid[:8]}: loading + compressing ...")
        t0 = time.perf_counter()
        r = bench_one(pid, one)
        results[pid] = r
        elapsed = time.perf_counter() - t0
        print(f"{pid[:8]}: n={r['n_spikes']:,} blosc={r['cr_blosc']:.1f}x zarr={r['cr_zarr']:.1f}x ({elapsed:.1f}s)")
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(list(results.values()), indent=2))

    print(f"wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
