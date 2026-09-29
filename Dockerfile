FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    git \
    curl \
    python3-pip \
    libzmq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir "pip<24.1" && \
    python3 -m pip install --no-cache-dir "setuptools==65.5.0" "wheel==0.38.4" && \
    python3 -m pip install --no-cache-dir --no-build-isolation gym==0.21.0 && \
    python3 -m pip install --no-cache-dir -r requirements.txt