"""
Integration tests for server.py — the FastMCP ``process_brain_file`` tool.

We call the tool function directly (unit style) rather than standing up a
full MCP transport, keeping the tests fast and dependency-free.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from brain_processor import _ELECTRON_ENERGY_KEV, _PHOTON_ENERGY_KEV
from server import process_brain_file


def _b64_numpy(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 for use as tool input."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


class TestProcessBrainFileTool:
    def test_basic_3d(self):
        rng = np.random.default_rng(0)
        arr = rng.random((4, 6, 6))
        result = process_brain_file(
            file_content=_b64_numpy(arr),
            file_format="numpy",
        )
        inner = result["data:3d"]
        assert inner["height"] == 4
        assert inner["amount_positions"] == 4 * 6 * 6
        emap = inner["energetic_map"]
        assert len(emap["Elektron"]) == 4 * 6 * 6
        assert len(emap["Photon"]) == 4 * 6 * 6

    def test_auto_detect_numpy(self):
        arr = np.ones((2, 3, 4))
        result = process_brain_file(file_content=_b64_numpy(arr))
        assert "data:3d" in result

    def test_invalid_base64_raises(self):
        with pytest.raises(ValueError, match="base64"):
            process_brain_file(file_content="not-valid-base64!!!")

    def test_2d_array_wrapped(self):
        arr = np.eye(4)
        result = process_brain_file(
            file_content=_b64_numpy(arr),
            file_format="numpy",
        )
        assert result["data:3d"]["height"] == 1

    def test_energetic_map_values_in_range(self):
        rng = np.random.default_rng(1)
        arr = rng.random((3, 5, 5))
        result = process_brain_file(
            file_content=_b64_numpy(arr),
            file_format="numpy",
        )
        emap = result["data:3d"]["energetic_map"]
        for val in emap["Elektron"].values():
            assert 0.0 <= val[0][0] <= _ELECTRON_ENERGY_KEV
        for val in emap["Photon"].values():
            assert 0.0 <= val[0][0] <= _PHOTON_ENERGY_KEV

    def test_constant_array_energetic_strength_is_zero(self):
        arr = np.full((2, 4, 4), 7.0)
        result = process_brain_file(
            file_content=_b64_numpy(arr),
            file_format="numpy",
        )
        emap = result["data:3d"]["energetic_map"]
        for channel in ("Elektron", "Photon"):
            for val in emap[channel].values():
                assert val[0][0] == pytest.approx(0.0)
                assert val[1][0] == pytest.approx(0.0)

    def test_none_file_format_uses_autodetect(self):
        arr = np.zeros((2, 4, 4))
        result = process_brain_file(
            file_content=_b64_numpy(arr),
            file_format=None,
        )
        assert "data:3d" in result

    def test_file_info_in_response(self):
        arr = np.ones((2, 3, 4))
        b64 = _b64_numpy(arr)
        result = process_brain_file(file_content=b64, file_format="numpy")
        info = result["data:3d"]["file_info"]
        assert info["format"] == "numpy"
        assert info["size_bytes"] > 0
        assert info["original_shape"] == [2, 3, 4]
        assert "dtype" in info
