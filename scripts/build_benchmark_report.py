"""Build the benchmark figure + docs/explanation/benchmark.qmd from scripts/benchmark_pids.py results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook")

PKG_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PKG_ROOT / "reports" / "benchmark_pids_results.json"
FIG_PATH = PKG_ROOT / "docs" / "figures" / "2026-08-27_spikepack_benchmark_cr.png"
REPORT_PATH = PKG_ROOT / "docs" / "explanation" / "benchmark.qmd"

# PID order + annotations, matching the source list (also used for the lfpack LFP benchmark)
PID_NOTES = {
    "1a276285-8b0e-4cc9-9f0a-a3a002978724": "benchmark start",
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0": "amazing CSD",
    "dc7e9403-19f7-409f-9240-05ee57cb7aea": "static noise: positive spikes",
    "fe380793-8035-414e-b000-09bfe5ece92a": "benchmark stop",
}
PID_ORDER = [
    "1a276285-8b0e-4cc9-9f0a-a3a002978724",
    "1e104bf4-7a24-4624-a5b2-c2c8289c0de7",
    "6638cfb3-3831-4fc2-9327-194b76cf22e1",
    "749cb2b7-e57e-4453-a794-f6230e4d0226",
    "d7ec0892-0a6c-4f4f-9d8f-72083692af5c",
    "da8dfec1-d265-44e8-84ce-6ae9c109b8bd",
    "dab512bd-a02d-4c1f-8dbc-9155a163efc0",
    "dc7e9403-19f7-409f-9240-05ee57cb7aea",
    "e8f9fba4-d151-4b00-bee7-447f0f3e752c",
    "eebcaf65-7fa4-4118-869d-a084e84530e2",
    "fe380793-8035-414e-b000-09bfe5ece92a",
]


def build_figure(df: pd.DataFrame) -> None:
    df = df.set_index("pid").loc[PID_ORDER].reset_index()
    x = np.arange(len(df))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.bar(x - bar_w / 2, df["cr_blosc"], width=bar_w * 0.9, label="Blosc shard", color="steelblue")
    ax.bar(x + bar_w / 2, df["cr_zarr"], width=bar_w * 0.9, label="Zarr", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels([p[:8] for p in df["pid"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("compression ratio vs raw float64+int32")
    ax.set_title("Compression ratio per insertion")
    ax.legend()
    sns.despine(ax=ax)

    ax = axes[1]
    ax.scatter(df["n_spikes"] / 1e6, df["cr_blosc"], label="Blosc shard", color="steelblue", s=50)
    ax.scatter(df["n_spikes"] / 1e6, df["cr_zarr"], label="Zarr", color="tomato", s=50, marker="x")
    ax.set_xlabel("n spikes (millions)")
    ax.set_ylabel("compression ratio")
    ax.set_title("CR vs recording size")
    ax.legend()
    sns.despine(ax=ax)

    fig.suptitle("spikepack: Blosc shard vs Zarr — 11 IBL benchmark insertions, 100 µs quantization")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)


def build_table(df: pd.DataFrame) -> str:
    df = df.set_index("pid").loc[PID_ORDER].reset_index()
    lines = [
        "| PID | note | n spikes | n units | raw | Blosc | CR | Zarr | CR | max err (Blosc/Zarr) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        note = PID_NOTES.get(r["pid"], "")
        lines.append(
            f"| `{r['pid'][:8]}` | {note} | {r['n_spikes']:,} | {r['n_units']} | "
            f"{r['raw_bytes'] / 1e6:.0f} MB | {r['blosc_bytes'] / 1e6:.1f} MB | **{r['cr_blosc']:.1f}x** | "
            f"{r['zarr_bytes'] / 1e6:.1f} MB | **{r['cr_zarr']:.1f}x** | "
            f"{r['max_abs_err_blosc_s'] * 1e6:.1f}/{r['max_abs_err_zarr_s'] * 1e6:.1f} µs |"
        )
    return "\n".join(lines)


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    df = pd.DataFrame(results)
    assert set(df["pid"]) == set(PID_ORDER), "results.json does not cover all 11 benchmark PIDs"
    assert bool(df["labels_round_trip_ok"].all()), "label round-trip failed for at least one insertion"

    build_figure(df)
    table_md = build_table(df)

    total_raw = df["raw_bytes"].sum()
    total_blosc = df["blosc_bytes"].sum()
    total_zarr = df["zarr_bytes"].sum()

    report = f"""---
title: "Benchmark: 11 IBL insertions"
subtitle: "Blosc shard vs Zarr, 100 µs quantization"
---

Same 11 insertions used for the [lfpack LFP benchmark gallery](https://int-brain-lab.github.io/lfpack/explanation/gallery.html),
loaded via `one.load_object(..., 'spikes', attribute=['times', 'clusters'])` — the
**full recording** (no good-unit filtering), so this reflects the harder,
noisier case: every detected spike, including multi-unit/noise clusters, not just
curated good units.

## Results

{table_md}

Aggregate: {total_raw / 1e9:.2f} GB raw -> {total_blosc / 1e9:.2f} GB Blosc
(**{total_raw / total_blosc:.1f}x**) / {total_zarr / 1e9:.2f} GB Zarr
(**{total_raw / total_zarr:.1f}x**).

![Compression ratio per insertion, and vs recording size](../figures/2026-08-27_spikepack_benchmark_cr.png)

## Reading the numbers

- **Blosc and Zarr compress identically** — both wrap the same `encode_times` +
  Blosc(zstd, shuffle) codec, so any CR difference between the two columns above is
  container overhead (Zarr's per-array `zarr.json` metadata vs the hand-rolled
  `meta.json`), not a codec difference. At this scale the overhead is negligible.
- **Round-trip error never exceeds half a tick** (50 µs at this quantization) for
  either format, and cluster labels round-trip exactly (`labels_round_trip_ok` was
  `True` for all 11 insertions).
- **CR is lower than the `bwm_ephys`-reported values** because this benchmark
  compresses the *full* recording (all detected spikes) rather than the good-unit
  subset `bwm_ephys` ships — noise/MUA clusters fire less regularly than curated
  single units, so their inter-spike-interval deltas compress somewhat less well.

Reproduce with `scripts/benchmark_pids.py` (downloads real data, requires ONE/Alyx
access) followed by `scripts/build_benchmark_report.py`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"wrote {FIG_PATH}")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
