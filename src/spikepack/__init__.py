"""spikepack — delta-quantization codec for sorted spike-time (event-time) arrays."""

from spikepack._core import (
    DEFAULT_QUANTIZATION_US,
    SpikepackError,
    compress_array,
    decode_times,
    decompress_array,
    encode_times,
)
from spikepack._io import read_blosc, read_zarr, write_blosc, write_zarr

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_QUANTIZATION_US",
    "SpikepackError",
    "compress_array",
    "decode_times",
    "decompress_array",
    "encode_times",
    "read_blosc",
    "read_zarr",
    "write_blosc",
    "write_zarr",
]
