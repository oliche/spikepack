"""Full-blown round-trip tests against real IBL spike trains.

Marked `network`: requires live ONE/Alyx access and downloads real data, so it is
deselected by default in CI (`pytest -m "not network"`). Run explicitly with
`pytest -m network` (or `pytest tests/test_benchmark_pids.py`).

This exercises the same 11 standard IBL benchmark insertions used for the lfpack
benchmark gallery, so results are directly comparable across codecs. The full
compression-ratio sweep with report generation lives in `scripts/benchmark_pids.py`;
this file only asserts correctness (round-trip fidelity, label integrity, a sane
compression floor) on a couple of them, to keep routine `-m network` runs fast.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("one.api")
pytest.importorskip("brainbox.io.one")

from spikepack import read_blosc, read_zarr, write_blosc, write_zarr  # noqa: E402

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

QUANTIZATION_US = 100


@pytest.fixture(scope="module")
def one():
    from one.api import ONE

    return ONE()


def _load_spikes(pid, one):
    from brainbox.io.one import SpikeSortingLoader

    ssl = SpikeSortingLoader(pid=pid, one=one)
    spikes = one.load_object(
        ssl.eid, "spikes", collection=f"alf/{ssl.pname}/pykilosort", attribute=["times", "clusters"]
    )
    times = np.asarray(spikes["times"], dtype=np.float64)
    clusters = np.asarray(spikes["clusters"], dtype=np.int32)
    return times, clusters


@pytest.mark.network
@pytest.mark.parametrize("pid", [PIDS[0], PIDS[6]])  # first + "amazing CSD" insertion
def test_real_insertion_round_trips_losslessly_in_order_and_labels(pid, one, tmp_path):
    times, clusters = _load_spikes(pid, one)
    assert times.size > 0
    assert np.all(np.diff(times) >= 0)

    write_blosc(tmp_path / "blosc_shard", times_seconds=times, labels=clusters, quantization_us=QUANTIZATION_US)
    blosc_out = read_blosc(tmp_path / "blosc_shard")
    np.testing.assert_array_equal(blosc_out["labels"], clusters)
    assert np.max(np.abs(blosc_out["times"] - times)) <= (QUANTIZATION_US * 1e-6) / 2 + 1e-9

    write_zarr(tmp_path / "zarr_store.zarr", times_seconds=times, labels=clusters, quantization_us=QUANTIZATION_US)
    zarr_out = read_zarr(tmp_path / "zarr_store.zarr")
    np.testing.assert_array_equal(zarr_out["labels"], clusters)
    np.testing.assert_array_equal(blosc_out["times"], zarr_out["times"])


@pytest.mark.network
@pytest.mark.parametrize("pid", [PIDS[0], PIDS[6]])
def test_real_insertion_compresses_well_below_raw_float64(pid, one, tmp_path):
    times, clusters = _load_spikes(pid, one)
    raw_bytes = times.nbytes + clusters.nbytes

    shard_bytes = write_blosc(
        tmp_path / "blosc_shard", times_seconds=times, labels=clusters, quantization_us=QUANTIZATION_US
    )
    assert shard_bytes < raw_bytes / 3  # real recordings comfortably clear the synthetic-data floor
