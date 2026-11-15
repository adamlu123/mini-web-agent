# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.11-slim-bookworm

ARG BENCHCAD_COMMIT
LABEL org.benchcad.commit="${BENCHCAD_COMMIT}"

COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglu1-mesa libgomp1 libosmesa6 libsm6 libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=benchcad pyproject.toml uv.lock /tmp/benchcad/
RUN cd /tmp/benchcad \
    && uv sync --frozen --no-dev \
    && mv .venv /opt/benchcad-venv \
    && rm -rf /tmp/benchcad

COPY --from=benchcad benchcad_core/__init__.py /opt/benchcad-tools/benchcad_core/
COPY --from=benchcad benchcad_core/scoring/__init__.py /opt/benchcad-tools/benchcad_core/scoring/
COPY --from=benchcad benchcad_core/scoring/exec_cq.py /opt/benchcad-tools/benchcad_core/scoring/
COPY --from=benchcad benchcad_core/scoring/views.py /opt/benchcad-tools/benchcad_core/scoring/
COPY src/cuawright/environments/benchcad/container/benchcad-run /usr/local/bin/benchcad-run
COPY src/cuawright/environments/benchcad/container/benchcad-render /usr/local/bin/benchcad-render
COPY --chown=0:0 --chmod=0555 \
    src/cuawright/environments/control_clients/binder.py \
    src/cuawright/environments/control_clients/critic.py \
    src/cuawright/environments/control_clients/verifier.py \
    src/cuawright/environments/control_clients/image_show.py \
    src/cuawright/environments/control_clients/ask_user.py \
    src/cuawright/environments/control_clients/submit.py \
    /opt/cuawright-tools/

RUN chmod 0755 /usr/local/bin/benchcad-run /usr/local/bin/benchcad-render \
    && useradd --create-home --uid 1000 --shell /bin/bash agent

ENV PATH="/opt/benchcad-venv/bin:${PATH}" \
    PYTHONPATH="/opt/benchcad-tools" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER agent
WORKDIR /workspace
