# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.1.0] - 2026-08-27

### Added
- Core codec: `encode_times` / `decode_times` — origin-normalized, quantized, delta-encoded integer ticks with automatic `uint16 -> uint32 -> uint64` dtype widening.
- `compress_array` / `decompress_array` — Blosc(zstd, shuffle) array codec.
- `write_blosc` / `read_blosc` — minimal directory-of-`.blosc`-files container (extracted from `bwm_ephys`'s spike shard format).
- `write_zarr` / `read_zarr` — Zarr v3 group container using the same Blosc(zstd, shuffle) codec.
- Extracted from `int-brain-lab/ibl-ai-agent`'s `bwm_ephys` spike-store builder (`delta_int_ticks + 100us + shuffle_zstd`, see `docs/decisions/bwm_ephys_spike_encoding.md` in that repo), generalized to arbitrary sorted event-time arrays with optional integer labels.
