# Dependency boundaries

CUAWright's base installation is intentionally dependency-free. It supports
importing `cuawright`, the core agent loop and runtime contracts, the prompt
registry, the configuration store, and CLI help/setup/doctor parsing without
loading provider or benchmark runtimes.

The maintained `src/cuawright` tree has the following non-standard-library
import roots:

| Boundary | Import evidence | Installation |
| --- | --- | --- |
| Current providers | `anthropic` (2 files), `openai` (2), `PIL` (3), `requests` (6) | `cuawright[providers]` |
| OSWorld | `desktop_env` (2), `task_loader` (1), plus the pinned external release's host/provider/evaluator requirements | `cuawright[osworld]` |
| WeaveBench | The pinned host runner declares `openai`, `huggingface-hub`, `tqdm`, `requests`, `gymnasium`, `docker`, `psutil`, `filelock`, `lxml`, and `flask`; the CUAWright adapter also imports the shared provider/image stack | `cuawright[weavebench]` in a dedicated environment |
| BenchCAD | `docker` (7 shared BenchCAD/CADGenBench files), `openai`, `PIL`, `requests`; `benchcad_core` is supplied by the pinned external checkout | `cuawright[benchcad]` |
| CADGenBench | `docker`, `mcp`, `pydantic`, `yaml`, `PIL`; `cadgenbench` is supplied by the pinned external checkout/container | `cuawright[cadgenbench]` |
| Odysseys | The same pinned OSWorld-compatible desktop host surface used by its generated examples; benchmark dataset and image judge remain external | `cuawright[odysseys]` |
| Inspection/media | Document, image, audio, OCR, browser, and ML inspection packages retained from the supported OSWorld tool/evaluator surface | `cuawright[inspection]` |
| Cloud/legacy | Cloud managers, historical provider SDKs, download clients, telemetry, and GitLab integration | `cuawright[cloud-legacy]` |
| Development | `pytest` and `black` | `uv sync --dev` |

The default OpenAI Responses endpoint is
`https://maif-gateway.azure-api.net/api`.
WeaveBench retains its fixed gateway-only condition.
Its adapter maps the pinned external launcher's legacy gateway URL to the new
gateway; the external checkout and official judge configuration are unchanged.

`cuawright[full]` duplicates the explicit compatible union for OSWorld,
BenchCAD, CADGenBench, Odysseys, inspection, providers, and retained cloud/legacy support.
It intentionally avoids self-referential extras, which are not portable across
all packaging tools. The broad OSWorld extra remains intentionally large
because the pinned external benchmark imports its provider, evaluator,
document/media, and platform-specific dependency surfaces at runtime.

WeaveBench is intentionally excluded from that single-runtime promise. Its
pinned commit requires `gymnasium>=0.29` and `tqdm>=4.66`; pinned OSWorld
requires `gymnasium~=0.28.1` and `tqdm~=4.65.0`. The `uv` metadata declares
`weavebench` incompatible with `osworld` and `full`, and CLI preflight rejects
the OSWorld versions with dedicated-environment guidance rather than claiming
the two can coexist.

The obsolete `dotenv` distribution is not retained: all repository
`from dotenv import ...` imports are provided by `python-dotenv`. `black` moved
from runtime dependencies to the development group. No other prior dependency
was dropped; packages not imported directly by maintained source are retained
in the external benchmark, inspection, cloud/legacy, or `full` sets.
