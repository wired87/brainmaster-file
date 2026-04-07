# brainmaster-file

> **Production MCP server** for processing research SPECT brain output files.  
> Receives a brain scan file, builds a 3-D voxel representation, and returns a
> per-voxel energetic strength map — ready to plug straight into any MCP-compatible
> client or agent framework.

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick start — Docker (recommended)](#quick-start--docker-recommended)
4. [Quick start — Local Python](#quick-start--local-python)
5. [MCP tool reference](#mcp-tool-reference)
6. [File format support](#file-format-support)
7. [Response schema](#response-schema)
8. [Running the tests](#running-the-tests)
9. [Project structure](#project-structure)

---

## Overview

`brainmaster-file` is a [FastMCP](https://github.com/jlowin/fastmcp) server that
exposes a single MCP tool: **`process_brain_file`**.

Processing pipeline:

```
Brain SPECT file (NIfTI / NumPy)
           │
           ▼
 Decode base64 → load volume
           │
           ▼
 Infer pixel count per 2-D slice
 → assemble 3-D volume (z × y × x)
           │
           ▼
 Normalise voxel intensities → [0, 1]
 (energetic strength per position)
           │
           ▼
 Return data:3d {height, amount_positions, energetic_map}
```

---

## Architecture

| File | Purpose |
|---|---|
| `server.py` | FastMCP server entry point; defines the `process_brain_file` tool |
| `brain_processor.py` | Core processing logic (format detection, loading, 3-D assembly, energetic map) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Multi-stage production Docker image (Python 3.12-slim, non-root user) |
| `docker-compose.yml` | Plug-and-play single-command startup |
| `tests/` | Unit + integration tests (pytest) |

---

## Quick start — Docker (recommended)

**Prerequisites:** Docker ≥ 20 and Docker Compose v2.

```bash
# 1. Clone the repo
git clone https://github.com/wired87/brainmaster-file.git
cd brainmaster-file

# 2. Build and start the server
docker compose up --build
```

The MCP server is now reachable at `http://localhost:8000`.

The container restarts automatically (`restart: unless-stopped`).

To stop:

```bash
docker compose down
```

---

## Quick start — Local Python

**Prerequisites:** Python ≥ 3.12.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Run with stdio transport (default — for local MCP clients)
python server.py

# 2b. Run with HTTP/SSE transport
python server.py --http --host 0.0.0.0 --port 8000
```

---

## MCP tool reference

### `process_brain_file`

Processes a brain SPECT output file and returns a 3-D energetic strength map.

#### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_content` | `string` | ✅ | Base64-encoded content of the brain scan file |
| `file_format` | `"nifti" \| "numpy" \| null` | ❌ | Explicit format hint. Omit for auto-detection |

#### Example call (Python client)

```python
import base64, nibabel as nib, numpy as np, io
from fastmcp import Client

# Load a NIfTI scan and encode it
img = nib.load("my_spect_scan.nii.gz")
buf = io.BytesIO()
img.to_file_map({"image": nib.FileHolder(fileobj=buf)})
b64 = base64.b64encode(buf.getvalue()).decode()

async with Client("http://localhost:8000/mcp") as client:
    result = await client.call_tool(
        "process_brain_file",
        {"file_content": b64, "file_format": "nifti"},
    )
    data = result[0].text  # JSON string
```

---

## File format support

| Format | Identifier | Extension | Library |
|---|---|---|---|
| NIfTI-1 / NIfTI-2 | `"nifti"` | `.nii`, `.nii.gz` | nibabel |
| NumPy binary | `"numpy"` | `.npy`, `.npz` | numpy |

When `file_format` is omitted the server auto-detects the format from magic
bytes (gzip header → NIfTI; NumPy magic → NumPy).

---

## Response schema

```json
{
  "data:3d": {
    "height": 128,
    "amount_positions": 8388608,
    "energetic_map": {
      "(0,0,0)": 0.234567,
      "(1,0,0)": 0.456789,
      "(0,1,0)": 0.123456,
      "...": "..."
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `height` | `int` | Number of z-slices (screens) in the 3-D volume |
| `amount_positions` | `int` | Total voxels: `height × rows × cols` |
| `energetic_map` | `dict[str, float]` | Energetic strength per voxel. Key format: `"(x,y,z)"`. Values are normalised to **[0, 1]** (0 = minimum intensity, 1 = maximum intensity) |

---

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Expected output: **33 passed**.

---

## Project structure

```
brainmaster-file/
├── brain_processor.py    # Core processing logic
├── server.py             # FastMCP server
├── requirements.txt      # Python dependencies
├── Dockerfile            # Production Docker image
├── docker-compose.yml    # Plug-and-play Docker Compose setup
├── .dockerignore
├── README.md
└── tests/
    ├── test_brain_processor.py   # Unit tests
    └── test_server.py            # Integration tests
```