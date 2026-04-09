"""
Brain file processor for research SPECT brain output files.

Accepts brain scan data (NIfTI or NumPy format), infers the number of pixels
in each 2D slice, assembles a 3D representation, and computes a per-voxel
energetic strength map suitable for downstream analysis.

The energetic map is split into two particle channels:

* **Elektron** — Elektron (electron) kinetic-energy proxy, scaled to 511 keV (electron
  rest-mass equivalent) at maximum voxel intensity.
* **Photon** — SPECT gamma-photon energy proxy, scaled to 140.5 keV (Tc-99m
  characteristic emission) at maximum voxel intensity.

For each voxel position ``(x, y, z)`` both channels carry::

    [[e_strength_keV], [frequency_Hz]]

where a single-element list is used to keep the schema extensible to
multi-component decompositions.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SPECT simulation)
# ---------------------------------------------------------------------------

# Elektron (electron) rest-mass energy (keV)
_ELECTRON_ENERGY_KEV: float = 511.0
# Frequency corresponding to electron rest-mass energy via E = h·f
# h = 6.626e-34 J·s;  511 keV = 511e3 × 1.602e-19 J
_ELECTRON_FREQ_HZ: float = 1.2356e20

# Tc-99m characteristic gamma-photon energy (keV) — standard SPECT tracer
_PHOTON_ENERGY_KEV: float = 140.5
# Frequency corresponding to 140.5 keV photon via E = h·f
_PHOTON_FREQ_HZ: float = 3.397e19


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _normalize(data: np.ndarray) -> np.ndarray:
    """Min-max normalise *data* to [0, 1].

    Returns an all-zero array when the data range is zero.
    """
    lo, hi = float(data.min()), float(data.max())
    if hi > lo:
        return (data.astype(np.float64) - lo) / (hi - lo)
    return np.zeros_like(data, dtype=np.float64)


def _ensure_3d(data: np.ndarray) -> np.ndarray:
    """Guarantee the array has exactly three dimensions.

    * 2-D input  → add a single z-slice: shape (1, rows, cols)
    * 3-D input  → returned unchanged
    * 4-D input  → first volume is used: shape (z, y, x)
    """
    if data.ndim == 2:
        return data[np.newaxis, ...]
    if data.ndim == 3:
        return data
    if data.ndim == 4:
        return data[..., 0]
    raise ValueError(
        f"Unsupported array dimensionality: {data.ndim}D. "
        "Expected 2-D, 3-D, or 4-D array."
    )


def _build_result(data: np.ndarray, file_info: dict | None = None) -> dict:
    """Convert a (potentially N-D) numpy array into the MCP response dict.

    The returned structure is::

        {
            "data:3d": {
                "height": <int>,           # number of z-slices (screens)
                "amount_positions": <int>, # total voxels in the volume
                "energetic_map": {
                    "Elektron": {          # Elektron (electron) energy channel
                        "(x,y,z)": [[e_strength_keV], [frequency_Hz]],
                        ...
                    },
                    "Photon": {            # gamma-photon energy channel
                        "(x,y,z)": [[e_strength_keV], [frequency_Hz]],
                        ...
                    }
                },
                "file_info": {             # metadata of the received file
                    "format": <str>,
                    "size_bytes": <int>,
                    "original_shape": [<int>, ...],
                    "dtype": <str>
                }
            }
        }

    Position keys use ``(x, y, z)`` convention where x is the column index
    and z is the slice index (screen number).

    Energy values are scaled by the normalised voxel intensity (0-1):
    * Elektron e_strength — up to ``_ELECTRON_ENERGY_KEV`` keV.
    * Photon   e_strength — up to ``_PHOTON_ENERGY_KEV``   keV.
    Frequencies follow E = h·f, giving Hz values proportional to energy.
    """
    data = _ensure_3d(data)

    # Axis layout after _ensure_3d: data[z, y, x]
    height = int(data.shape[0])          # number of z-slices / screens
    rows = int(data.shape[1])            # y dimension
    cols = int(data.shape[2])            # x dimension
    amount_positions = height * rows * cols

    normalised = _normalize(data)

    elektron_map: dict[str, list] = {}
    photon_map: dict[str, list] = {}

    for z in range(height):
        for y in range(rows):
            for x in range(cols):
                pos = f"({x},{y},{z})"
                intensity = round(float(normalised[z, y, x]), 6)

                elektron_map[pos] = [
                    [round(intensity * _ELECTRON_ENERGY_KEV, 6)],
                    [round(intensity * _ELECTRON_FREQ_HZ, 6)],
                ]
                photon_map[pos] = [
                    [round(intensity * _PHOTON_ENERGY_KEV, 6)],
                    [round(intensity * _PHOTON_FREQ_HZ, 6)],
                ]

    result: dict = {
        "data:3d": {
            "height": height,
            "amount_positions": amount_positions,
            "energetic_map": {
                "Elektron": elektron_map,
                "Photon": photon_map,
            },
        }
    }

    if file_info is not None:
        result["data:3d"]["file_info"] = file_info

    return result


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------

def _load_nifti(file_bytes: bytes) -> np.ndarray:
    """Load a NIfTI-1 / NIfTI-2 file from raw bytes and return its data array.

    Uses *nibabel* which supports `.nii` and `.nii.gz` (gzip-compressed).
    A temporary file is used because nibabel expects a seekable file path.
    """
    import nibabel as nib  # optional import – keeps cold-start fast

    suffix = ".nii.gz" if file_bytes[:2] == b"\x1f\x8b" else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        img = nib.load(tmp_path)
        data = np.asarray(img.dataobj, dtype=np.float64)
    finally:
        os.unlink(tmp_path)

    return data


def _load_numpy(file_bytes: bytes) -> np.ndarray:
    """Load a NumPy ``.npy`` / ``.npz`` file from raw bytes."""
    buf = io.BytesIO(file_bytes)
    loaded = np.load(buf, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        # Use the first array in the archive
        key = list(loaded.keys())[0]
        return loaded[key].astype(np.float64)
    return np.asarray(loaded, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class BrainFileProcessor:
    """Process research SPECT brain output files into 3-D energetic maps.

    Supported formats
    -----------------
    ``"nifti"``
        NIfTI-1 / NIfTI-2 files (`.nii`, `.nii.gz`).  Requires *nibabel*.
    ``"numpy"``
        NumPy binary files (`.npy`, `.npz`).

    Usage
    -----
    >>> processor = BrainFileProcessor()
    >>> result = processor.process(raw_bytes, file_format="nifti")
    >>> result["data:3d"]["height"]
    128
    """

    _LOADERS = {
        "nifti": _load_nifti,
        "numpy": _load_numpy,
    }

    @classmethod
    def supported_formats(cls) -> list[str]:
        """Return the list of supported file format identifiers."""
        return list(cls._LOADERS.keys())

    def detect_format(self, file_bytes: bytes) -> str:
        """Guess file format from magic bytes.

        Returns ``"nifti"`` for NIfTI (plain and gzip-compressed) and
        ``"numpy"`` for NumPy binary files.  Raises ``ValueError`` if the
        format cannot be determined.
        """
        if file_bytes[:2] == b"\x1f\x8b":
            return "nifti"   # gzip → likely .nii.gz
        if file_bytes[:2] == b"\x93N":
            return "numpy"   # NumPy magic header \x93NUMPY
        # Plain NIfTI: check for "ni1\0" or "n+1\0" at byte 344
        if len(file_bytes) > 348:
            magic = file_bytes[344:348]
            if magic in (b"ni1\x00", b"n+1\x00", b"ni2\x00", b"n+2\x00"):
                return "nifti"
        raise ValueError(
            "Cannot detect file format from magic bytes. "
            f"Please specify file_format explicitly. "
            f"Supported formats: {self.supported_formats()}"
        )

    def process(
        self,
        file_bytes: bytes,
        file_format: Union[str, None] = None,
    ) -> dict:
        """Process *file_bytes* and return the 3-D energetic map response.

        Parameters
        ----------
        file_bytes:
            Raw bytes of the brain scan file.
        file_format:
            One of ``"nifti"`` or ``"numpy"``.  When ``None`` the format is
            auto-detected from magic bytes.

        Returns
        -------
        dict
            ``{"data:3d": {"height": int, "amount_positions": int,
            "energetic_map": {"Elektron": {...}, "Photon": {...}},
            "file_info": {"format": str, "size_bytes": int,
            "original_shape": list[int], "dtype": str}}}``
        """
        if not file_bytes:
            raise ValueError("file_bytes must not be empty.")

        fmt = file_format.lower().strip() if file_format else self.detect_format(file_bytes)

        loader = self._LOADERS.get(fmt)
        if loader is None:
            raise ValueError(
                f"Unsupported file format: {fmt!r}. "
                f"Choose one of {self.supported_formats()}."
            )

        data = loader(file_bytes)

        file_info: dict = {
            "format": fmt,
            "size_bytes": len(file_bytes),
            "original_shape": list(data.shape),
            "dtype": str(data.dtype),
        }

        return _build_result(data, file_info=file_info)
