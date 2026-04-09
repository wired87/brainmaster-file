"""
BrainMaster File — FastMCP server.

Exposes a single MCP tool, ``process_brain_file``, that accepts a
base64-encoded brain SPECT output file and returns a structured 3-D
energetic map.

Run locally
-----------
    python server.py              # stdio transport (default)
    python server.py --http       # HTTP/SSE transport on port 8000

Docker
------
See ``docker-compose.yml`` for the plug-and-play setup.
"""

from __future__ import annotations

import argparse
import base64
import sys
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from brain_processor import BrainFileProcessor

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "BrainMaster File",
    instructions=(
        "Processes research SPECT brain output files. "
        "Call `process_brain_file` with a base64-encoded scan to receive "
        "the 3-D energetic strength map."
    ),
)

_processor = BrainFileProcessor()


# ---------------------------------------------------------------------------
# Health endpoint (used by Docker HEALTHCHECK)
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    """Lightweight liveness probe — always returns HTTP 200 when the server is up."""
    return JSONResponse({"status": "ok"})

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@mcp.tool()
def process_brain_file(
    file_content: Annotated[
        str,
        Field(
            description=(
                "Base64-encoded content of the brain SPECT output file. "
                "Supported formats: NIfTI (.nii / .nii.gz) and NumPy (.npy / .npz)."
            )
        ),
    ],
    file_format: Annotated[
        Literal["nifti", "numpy"] | None,
        Field(
            default=None,
            description=(
                "Explicit file format identifier. "
                "When omitted the format is auto-detected from the file's "
                "magic bytes. "
                "Accepted values: 'nifti', 'numpy'."
            ),
        ),
    ] = None,
) -> dict:
    """Process a research SPECT/PET brain output file and return a 3-D energetic map.

    The tool performs the following steps:

    1. Decodes the base64-encoded file content.
    2. Loads the brain scan volume (NIfTI or NumPy format).
    3. Infers the pixel count of each 2-D slice (screen) and assembles the
       full 3-D volume.
    4. Normalises each voxel's intensity to an energetic strength in [0, 1].
    5. Builds per-voxel **Elektron** and **Photon** energy/frequency channels.
    6. Returns the structured result as a JSON-serialisable dict.

    Returns
    -------
    A dict with a single key ``"data:3d"`` containing:

    * ``height`` — number of z-slices (screens) in the 3-D volume.
    * ``amount_positions`` — total number of voxels (height × rows × cols).
    * ``energetic_map`` — dict with two particle-type keys:

      * ``"Elektron"`` — Elektron (electron) energy channel.
        Each entry: ``"(x,y,z)": [[e_strength_keV], [frequency_Hz]]``
        Energy scaled to 511 keV (electron rest-mass) at max voxel intensity.
      * ``"Photon"`` — gamma-photon energy channel (Tc-99m SPECT).
        Each entry: ``"(x,y,z)": [[e_strength_keV], [frequency_Hz]]``
        Energy scaled to 140.5 keV at max voxel intensity.

    * ``file_info`` — metadata of the received file:
      ``format``, ``size_bytes``, ``original_shape``, ``dtype``.

    Example response
    ----------------
    ```json
    {
      "data:3d": {
        "height": 4,
        "amount_positions": 192,
        "energetic_map": {
          "Elektron": {
            "(0,0,0)": [[119.81], [2.98e+19]],
            "(1,0,0)": [[233.24], [5.81e+19]],
            "...": "..."
          },
          "Photon": {
            "(0,0,0)": [[32.94], [7.98e+18]],
            "(1,0,0)": [[64.04], [1.55e+19]],
            "...": "..."
          }
        },
        "file_info": {
          "format": "nifti",
          "size_bytes": 2097152,
          "original_shape": [128, 128, 4],
          "dtype": "float64"
        }
      }
    }
    ```
    """
    try:
        file_bytes = base64.b64decode(file_content, validate=True)
    except Exception as exc:
        raise ValueError(
            f"file_content is not valid base64: {exc}"
        ) from exc

    return _processor.process(file_bytes, file_format=file_format)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="brainmaster-file",
        description="BrainMaster File MCP server",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Start in HTTP/SSE mode instead of stdio (default)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind when --http is used (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind when --http is used (default: 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")
