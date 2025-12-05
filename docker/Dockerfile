# Use Ubuntu 24.04 with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /work

# Copy requirements and install Python dependencies
COPY requirements.txt /work/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /work/

# Set Python path
ENV PYTHONPATH=/work:$PYTHONPATH

# Create directories for frontend
RUN mkdir -p /work/frontend/uploads /work/frontend/outputs

# Expose port for web interface
EXPOSE 5000

# Default command (can be overridden)
CMD ["/bin/bash"]
