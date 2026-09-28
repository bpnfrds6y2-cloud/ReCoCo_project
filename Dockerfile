FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \ 
    python3.10 \ 
    python3.10-distutils \ 
    git \ 
    curl \ 
    python3-pip \ 
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt
