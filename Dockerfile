FROM python:3.11-slim-bookworm

ENV VIRTUAL_ENV=/opt/fly-miko-venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR=/tmp/fly-miko-numba \
    XDG_CACHE_HOME=/tmp/fly-miko-cache
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# The official Python image includes ensurepip. Keep the environment outside
# /workspace so mounting a copied project cannot replace it with a host venv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglfw3 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "${VIRTUAL_ENV}"

COPY requirements-sim.txt /tmp/requirements-sim.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-sim.txt \
    && python -m pip check

WORKDIR /workspace/fly-miko
COPY simulation/ ./simulation/
COPY config/ ./config/
COPY third_party/ ./third_party/
COPY NOTICE.md LICENSES.md ./
COPY scripts/prepare_brain_binding.py scripts/prepare_reproduction.py ./scripts/

# Check native MuJoCo loading and a physics step during the image build.
RUN python -m simulation.train --help \
    && python -c "from simulation.biped import Biped; b = Biped(); b.advance(dict(zip((j['id'] for j in b.joints), b.targets)))"

ENTRYPOINT ["python", "-m", "simulation.train"]
CMD ["--help"]
