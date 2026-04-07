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
    """Process a research SPECT brain output file and return a 3-D energetic map.

    The tool performs the following steps:

    1. Decodes the base64-encoded file content.
    2. Loads the brain scan volume (NIfTI or NumPy format).
    3. Infers the pixel count of each 2-D slice (screen) and assembles the
       full 3-D volume.
    4. Normalises each voxel's intensity to an energetic strength in [0, 1].
    5. Returns the structured result.

    Returns
    -------
    A dict with a single key ``"data:3d"`` containing:

    * ``height`` — number of z-slices (screens) in the 3-D volume.
    * ``amount_positions`` — total number of voxels (height × rows × cols).
    * ``energetic_map`` — direct ``dict[str, float]`` mapping every voxel
      position ``"(x,y,z)"`` to its normalised energetic strength value.

    Example response
    ----------------
    ```json
    {
      "data:3d": {
        "height": 128,
        "amount_positions": 8388608,
        "energetic_map": {
          "(0,0,0)": 0.234567,
          "(1,0,0)": 0.456789,
          ...
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
