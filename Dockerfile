FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /home
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6
WORKDIR /home
RUN pip install -r requirements.txt