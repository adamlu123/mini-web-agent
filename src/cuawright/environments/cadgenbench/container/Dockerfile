# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ARG CADGENBENCH_COMMIT
ARG BUILD123D_MCP_VERSION=0.3.59
LABEL org.cadgenbench.commit="${CADGENBENCH_COMMIT}"
LABEL org.build123d-mcp.version="${BUILD123D_MCP_VERSION}"
LABEL org.build123d.version="0.11.0"
LABEL org.build123d-drafting-helpers.version="0.12.0"
LABEL org.bd-warehouse.version="0.2.0"
LABEL org.augura.version="0.1.5"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fontconfig libegl1 libgl1 libglib2.0-0 libglu1-mesa libgomp1 libosmesa6 \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cadgenbench . /opt/cadgenbench-source
RUN python -m venv /opt/cadgenbench-venv \
    && /opt/cadgenbench-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/cadgenbench-venv/bin/pip install --no-cache-dir /opt/cadgenbench-source \
    && /opt/cadgenbench-venv/bin/pip install --no-cache-dir "mcp>=1.9,<2" \
    && /opt/cadgenbench-venv/bin/pip install --no-cache-dir \
        "build123d-mcp[http]==${BUILD123D_MCP_VERSION}" \
        "build123d==0.11.0" \
        "build123d-drafting-helpers==0.12.0" \
        "bd_warehouse==0.2.0" \
        "augura==0.1.5"

COPY src/cuawright/__init__.py /opt/cuawright/cuawright/__init__.py
COPY src/cuawright/environments/__init__.py /opt/cuawright/cuawright/environments/__init__.py
COPY src/cuawright/environments/cadgenbench /opt/cuawright/cuawright/environments/cadgenbench
COPY src/cuawright/environments/cadgenbench/container/cadgenbench-run /usr/local/bin/cadgenbench-run
COPY src/cuawright/environments/cadgenbench/container/cadgenbench-validate /usr/local/bin/cadgenbench-validate
COPY src/cuawright/environments/cadgenbench/container/cadgenbench-render /usr/local/bin/cadgenbench-render
COPY src/cuawright/environments/cadgenbench/container/cad-mcp /usr/local/bin/cad-mcp
COPY --chown=0:0 --chmod=0555 \
    src/cuawright/environments/control_clients/binder.py \
    src/cuawright/environments/control_clients/critic.py \
    src/cuawright/environments/control_clients/verifier.py \
    src/cuawright/environments/control_clients/image_show.py \
    src/cuawright/environments/control_clients/ask_user.py \
    src/cuawright/environments/control_clients/submit.py \
    /opt/cuawright-tools/

RUN chmod 0755 \
        /usr/local/bin/cad-mcp \
        /usr/local/bin/cadgenbench-run \
        /usr/local/bin/cadgenbench-validate \
        /usr/local/bin/cadgenbench-render \
    && useradd --create-home --uid 1000 --shell /bin/bash agent

ENV PATH="/opt/cadgenbench-venv/bin:${PATH}" \
    PYTHONPATH="/opt/cuawright:/opt/cadgenbench-source/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYVISTA_OFF_SCREEN=true \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CADGENBENCH_OCC_PARALLEL_MESH=0 \
    XDG_CACHE_HOME=/home/agent/.cache \
    MPLCONFIGDIR=/home/agent/.cache/matplotlib

USER agent
WORKDIR /workspace
