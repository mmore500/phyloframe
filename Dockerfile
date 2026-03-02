FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TQDM_MININTERVAL=5

RUN apt-get update \
    && apt-get install -y --no-install-recommends  \
        build-essential \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 --version

COPY . /app

RUN python3 -m pip install uv --break-system-packages \
    && python3 -m uv pip install --system --break-system-packages "/app"

# Clean up
RUN apt-get clean \
    && rm -rf /root/.cache /tmp/* /app

ENTRYPOINT echo "Error:  no default entrypoint; use 'python3 -m phyloframe' instead." >&2; exit 1
