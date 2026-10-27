FROM python:3.12.2-slim-bookworm@sha256:5dc6f84b5e97bfb0c90abfb7c55f3cacc2cb6687c8f920b64a833a2219875997

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TZ=UTC \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

WORKDIR /workspace

COPY reproducibility/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
      --require-hashes -r /tmp/requirements.lock

COPY . /workspace

# The wrapper runs as the invoking host UID so mounted artifacts are not root-owned.
# These are the only image-local paths changed during reproduction.
RUN chmod -R a+rwX /workspace/results /workspace/figures \
    && chmod +x /workspace/reproducibility/run_locked.sh

ENTRYPOINT ["/workspace/reproducibility/run_locked.sh"]
