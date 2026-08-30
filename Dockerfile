# syntax=docker/dockerfile:1
#
# Base choice: python:3.10-slim rather than an nvidia/cuda image.
#
# ctranslate2 4.7.1 needs cuDNN 9, and the nvidia/cuda *-cudnn-runtime tags only
# carry cuDNN 9 from CUDA 12.4 onwards. That would work on the target's 535
# driver through minor-version compatibility, but it would mean *relying* on
# that guarantee. Installing the same CUDA wheels the production venv already
# uses ships a byte-identical library set to something known to work on this
# exact hardware, and produces a much smaller image.
#
# libcuda.so is injected at runtime by the NVIDIA container toolkit.

# ---------- build ----------
FROM python:3.10-slim-bookworm AS builder

# No apt cache mount: both stages would share one cache, BuildKit runs them
# concurrently, and they deadlock on /var/cache/apt/archives/lock. The layer
# cache already makes rebuilds fast.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r /tmp/requirements.txt

# ---------- runtime ----------
FROM python:3.10-slim-bookworm

# ffmpeg/ffprobe are used ONLY by the remux repair path - audio decoding is done
# in-process by PyAV. Kept because that path has demonstrably rescued files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Put the CUDA wheel libraries on the loader path permanently.
#
# The systemd deployment this replaces set LD_LIBRARY_PATH to a hardcoded venv
# path containing the Python minor version, so a 3.10 -> 3.11 bump silently
# broke CUDA and the app fell back to CPU at 20-50x slower. Baking the paths
# into ld.so.conf means the linker finds them regardless of how the process is
# started, and nothing outside this image can break it.
RUN set -eux; \
    SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    printf '%s/nvidia/cublas/lib\n%s/nvidia/cudnn/lib\n' "$SITE" "$SITE" \
        > /etc/ld.so.conf.d/nvidia-wheels.conf; \
    ldconfig

# Fail the BUILD if the CUDA wiring is wrong, so a broken image can never reach
# the host. Importing ctranslate2 resolves its shared libraries; ldd with no
# "not found" lines proves every dependency is on the path.
RUN set -eux; \
    python -c "import ctranslate2, faster_whisper; print('ctranslate2', ctranslate2.__version__)"; \
    SO="$(python -c 'import ctranslate2, pathlib; print(pathlib.Path(ctranslate2.__file__).parent)')"; \
    ! ldd "$SO"/*.so* 2>/dev/null | grep -q 'not found'

# uid/gid 1000 matches the NFS export mapping on the target and the other
# containers on that host, so output lands readable by Plex and Stash.
RUN groupadd --gid 1000 app && useradd --uid 1000 --gid 1000 --create-home app

COPY --chown=1000:1000 src/ /app/src/
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model cache. A volume, not baked in: 3 GB of rarely-changing weights does
    # not belong in every image pull.
    HF_HOME=/models \
    # SQLite lives here. MUST be a local volume - SQLite over NFS corrupts.
    SW_CONFIG_DIR=/config \
    # Required on a non-nvidia base image; the toolkit reads these.
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # ctranslate2 spawns a thread per core by default. This host runs a dozen
    # other containers, so leave it some room.
    OMP_NUM_THREADS=4

RUN mkdir -p /models /config && chown 1000:1000 /models /config

USER 1000:1000
WORKDIR /app
EXPOSE 8420

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8420/healthz || exit 1

ENTRYPOINT ["python", "-m", "subwright"]

ARG VERSION=dev
ARG REVISION=unknown
LABEL org.opencontainers.image.title="subwright" \
      org.opencontainers.image.description="Watches a folder and generates English subtitles for video using Whisper on a GPU." \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="MIT"
