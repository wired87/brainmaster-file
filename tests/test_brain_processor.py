"""
Unit tests for brain_processor.py.

Tests cover:
- _normalize
- _ensure_3d
- _build_result  (Elektron/Photon energetic maps + file_info)
- BrainFileProcessor.detect_format
- BrainFileProcessor.process with NumPy format
- BrainFileProcessor.process with NIfTI format
- Edge cases (empty data, constant data, 4-D input)
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from brain_processor import (
    BrainFileProcessor,
    _ELECTRON_ENERGY_KEV,
    _ELECTRON_FREQ_HZ,
    _PHOTON_ENERGY_KEV,
    _PHOTON_FREQ_HZ,
    _build_result,
    _ensure_3d,
    _normalize,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_range_0_to_1(self):
        arr = np.array([0.0, 1.0, 2.0, 4.0])
        result = _normalize(arr)
        assert float(result.min()) == pytest.approx(0.0)
        assert float(result.max()) == pytest.approx(1.0)

    def test_constant_array_returns_zeros(self):
        arr = np.full((3, 3), 5.0)
        result = _normalize(arr)
        assert (result == 0.0).all()

    def test_single_element(self):
        arr = np.array([42.0])
        result = _normalize(arr)
        assert float(result[0]) == pytest.approx(0.0)

    def test_dtype_float64(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _normalize(arr)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# _ensure_3d
# ---------------------------------------------------------------------------

class TestEnsure3d:
    def test_2d_adds_z_axis(self):
        arr = np.zeros((4, 5))
        result = _ensure_3d(arr)
        assert result.shape == (1, 4, 5)

    def test_3d_unchanged(self):
        arr = np.zeros((3, 4, 5))
        result = _ensure_3d(arr)
        assert result.shape == (3, 4, 5)

    def test_4d_uses_first_volume(self):
        arr = np.zeros((3, 4, 5, 2))
        result = _ensure_3d(arr)
        assert result.shape == (3, 4, 5)

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="dimensionality"):
            _ensure_3d(np.zeros((10,)))


# ---------------------------------------------------------------------------
# _build_result
# ---------------------------------------------------------------------------

class TestBuildResult:
    def _make_data(self, shape=(2, 3, 4)):
        rng = np.random.default_rng(0)
        return rng.random(shape)

    def test_output_keys(self):
        data = self._make_data()
        result = _build_result(data)
        assert "data:3d" in result
        inner = result["data:3d"]
        assert "height" in inner
        assert "amount_positions" in inner
        assert "energetic_map" in inner

    def test_height_matches_z(self):
        data = self._make_data((5, 4, 3))
        inner = _build_result(data)["data:3d"]
        assert inner["height"] == 5

    def test_amount_positions(self):
        z, y, x = 5, 4, 3
        data = self._make_data((z, y, x))
        inner = _build_result(data)["data:3d"]
        assert inner["amount_positions"] == z * y * x

    def test_energetic_map_has_elektron_and_photon(self):
        data = self._make_data((2, 3, 4))
        emap = _build_result(data)["data:3d"]["energetic_map"]
        assert "Elektron" in emap
        assert "Photon" in emap

    def test_energetic_map_length(self):
        z, y, x = 2, 3, 4
        data = self._make_data((z, y, x))
        emap = _build_result(data)["data:3d"]["energetic_map"]
        assert len(emap["Elektron"]) == z * y * x
        assert len(emap["Photon"]) == z * y * x

    def test_energetic_map_key_format(self):
        data = self._make_data((1, 2, 3))
        emap = _build_result(data)["data:3d"]["energetic_map"]
        for channel in ("Elektron", "Photon"):
            for key in emap[channel]:
                assert key.startswith("(") and key.endswith(")")
                parts = key[1:-1].split(",")
                assert len(parts) == 3
                assert all(p.isdigit() for p in parts)

    def test_energetic_map_entry_structure(self):
        """Each entry must be [[e_strength], [frequency]] — two single-element lists."""
        data = self._make_data((1, 2, 2))
        emap = _build_result(data)["data:3d"]["energetic_map"]
        for channel in ("Elektron", "Photon"):
            for val in emap[channel].values():
                assert isinstance(val, list) and len(val) == 2
                e_strengths, frequencies = val
                assert isinstance(e_strengths, list) and len(e_strengths) == 1
                assert isinstance(frequencies, list) and len(frequencies) == 1

    def test_elektron_energy_range(self):
        """Elektron e_strength must be in [0, _ELECTRON_ENERGY_KEV]."""
        data = self._make_data((3, 4, 5))
        emap = _build_result(data)["data:3d"]["energetic_map"]["Elektron"]
        for val in emap.values():
            assert 0.0 <= val[0][0] <= _ELECTRON_ENERGY_KEV

    def test_photon_energy_range(self):
        """Photon e_strength must be in [0, _PHOTON_ENERGY_KEV]."""
        data = self._make_data((3, 4, 5))
        emap = _build_result(data)["data:3d"]["energetic_map"]["Photon"]
        for val in emap.values():
            assert 0.0 <= val[0][0] <= _PHOTON_ENERGY_KEV

    def test_elektron_frequency_range(self):
        """Elektron frequency must be in [0, _ELECTRON_FREQ_HZ]."""
        data = self._make_data((2, 3, 4))
        emap = _build_result(data)["data:3d"]["energetic_map"]["Elektron"]
        for val in emap.values():
            assert 0.0 <= val[1][0] <= _ELECTRON_FREQ_HZ

    def test_photon_frequency_range(self):
        """Photon frequency must be in [0, _PHOTON_FREQ_HZ]."""
        data = self._make_data((2, 3, 4))
        emap = _build_result(data)["data:3d"]["energetic_map"]["Photon"]
        for val in emap.values():
            assert 0.0 <= val[1][0] <= _PHOTON_FREQ_HZ

    def test_constant_array_all_zero_energy(self):
        """Constant intensity → min-max normalises to 0 → all energies are 0."""
        data = np.ones((2, 3, 3)) * 5.0
        emap = _build_result(data)["data:3d"]["energetic_map"]
        for channel in ("Elektron", "Photon"):
            for val in emap[channel].values():
                assert val[0][0] == pytest.approx(0.0)
                assert val[1][0] == pytest.approx(0.0)

    def test_file_info_included_when_provided(self):
        data = self._make_data((2, 3, 4))
        info = {"format": "numpy", "size_bytes": 100, "original_shape": [2, 3, 4], "dtype": "float64"}
        inner = _build_result(data, file_info=info)["data:3d"]
        assert "file_info" in inner
        assert inner["file_info"]["format"] == "numpy"
        assert inner["file_info"]["size_bytes"] == 100

    def test_file_info_absent_when_not_provided(self):
        data = self._make_data((2, 3, 4))
        inner = _build_result(data)["data:3d"]
        assert "file_info" not in inner

    def test_2d_input_wrapped(self):
        data = np.ones((4, 5)) * 3
        inner = _build_result(data)["data:3d"]
        assert inner["height"] == 1
        assert inner["amount_positions"] == 4 * 5


# ---------------------------------------------------------------------------
# BrainFileProcessor
# ---------------------------------------------------------------------------

class TestBrainFileProcessor:
    processor = BrainFileProcessor()

    # -- supported_formats --------------------------------------------------

    def test_supported_formats_contains_nifti_and_numpy(self):
        fmts = self.processor.supported_formats()
        assert "nifti" in fmts
        assert "numpy" in fmts

    # -- detect_format ------------------------------------------------------

    def test_detect_numpy(self):
        arr = np.zeros((3, 4, 5))
        buf = io.BytesIO()
        np.save(buf, arr)
        raw = buf.getvalue()
        assert self.processor.detect_format(raw) == "numpy"

    def test_detect_nifti_gzip(self):
        # First two bytes are the gzip magic
        fake_gz = b"\x1f\x8b" + b"\x00" * 100
        assert self.processor.detect_format(fake_gz) == "nifti"

    def test_detect_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot detect"):
            self.processor.detect_format(b"\x00\x01\x02\x03" * 100)

    # -- process with NumPy format ------------------------------------------

    def _numpy_bytes(self, arr: np.ndarray) -> bytes:
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    def test_process_numpy_3d(self):
        rng = np.random.default_rng(42)
        arr = rng.random((4, 8, 8))
        raw = self._numpy_bytes(arr)
        result = self.processor.process(raw, file_format="numpy")
        inner = result["data:3d"]
        assert inner["height"] == 4
        assert inner["amount_positions"] == 4 * 8 * 8
        assert len(inner["energetic_map"]["Elektron"]) == 4 * 8 * 8
        assert len(inner["energetic_map"]["Photon"]) == 4 * 8 * 8

    def test_process_numpy_2d(self):
        arr = np.eye(5)
        raw = self._numpy_bytes(arr)
        result = self.processor.process(raw, file_format="numpy")
        inner = result["data:3d"]
        assert inner["height"] == 1
        assert inner["amount_positions"] == 5 * 5

    def test_process_auto_detect_numpy(self):
        arr = np.ones((2, 3, 4))
        raw = self._numpy_bytes(arr)
        # No file_format → auto-detect
        result = self.processor.process(raw)
        assert "data:3d" in result

    def test_process_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.processor.process(b"")

    def test_process_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.processor.process(b"\x00" * 100, file_format="dicom")

    def test_energetic_map_values_in_range(self):
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 1000, size=(3, 5, 5)).astype(np.float64)
        raw = self._numpy_bytes(arr)
        result = self.processor.process(raw, file_format="numpy")
        emap = result["data:3d"]["energetic_map"]
        for channel, max_energy, max_freq in (
            ("Elektron", _ELECTRON_ENERGY_KEV, _ELECTRON_FREQ_HZ),
            ("Photon", _PHOTON_ENERGY_KEV, _PHOTON_FREQ_HZ),
        ):
            for val in emap[channel].values():
                assert 0.0 <= val[0][0] <= max_energy
                assert 0.0 <= val[1][0] <= max_freq

    def test_process_file_info_present(self):
        arr = np.ones((2, 3, 4))
        raw = self._numpy_bytes(arr)
        result = self.processor.process(raw, file_format="numpy")
        info = result["data:3d"]["file_info"]
        assert info["format"] == "numpy"
        assert info["size_bytes"] == len(raw)
        assert info["original_shape"] == [2, 3, 4]
        assert "dtype" in info

    # -- process with NIfTI format ------------------------------------------

    def test_process_nifti(self):
        """Create a minimal NIfTI-1 file in-memory and process it."""
        import nibabel as nib

        rng = np.random.default_rng(99)
        arr = rng.random((8, 8, 4)).astype(np.float32)
        img = nib.Nifti1Image(arr, affine=np.eye(4))
        buf = io.BytesIO()
        img.to_file_map({"image": nib.FileHolder(fileobj=buf)})
        raw = buf.getvalue()

        result = self.processor.process(raw, file_format="nifti")
        inner = result["data:3d"]
        # NIfTI stores axes as (x, y, z); after _ensure_3d the z-axis maps
        # to height.  The exact shape depends on how nibabel loads the data.
        assert inner["height"] > 0
        assert inner["amount_positions"] > 0
        emap = inner["energetic_map"]
        assert "Elektron" in emap
        assert "Photon" in emap
        assert isinstance(emap["Elektron"], dict)
        assert isinstance(emap["Photon"], dict)
        # file_info must be present and carry nifti format
        assert inner["file_info"]["format"] == "nifti"
