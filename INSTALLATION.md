# Autotune-AI Installation Guide

Complete installation instructions for all platforms and use cases.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Docker Installation](#docker-installation)
4. [Local Installation](#local-installation)
5. [GPU Setup](#gpu-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Ubuntu 24.04 LTS (or compatible Linux distribution)
- **Python**: 3.12 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space for installation and data
- **Audio**: FFmpeg with libsndfile1

### Recommended for GPU Acceleration

- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+
- **CUDA**: 12.2 or higher
- **cuDNN**: Compatible version with CUDA
- **VRAM**: 4GB+ (8GB recommended for model training)

### Performance Expectations

**With GPU (NVIDIA RTX 3080):**
- 3-minute song processing: ~2 minutes
- Batch processing (10 songs): ~5 minutes (with 4 parallel jobs)

**With CPU (Intel i7):**
- 3-minute song processing: ~10-15 minutes
- Batch processing (10 songs): ~30-40 minutes

## Quick Start

### Option 1: One-Line Quick Start (Linux)

For the fastest setup with the web interface:

```bash
curl -fsSL https://raw.githubusercontent.com/groxaxo/autotune-ai/main/run_frontend.sh | bash
```

Or manually:

```bash
git clone https://github.com/groxaxo/autotune-ai.git
cd autotune-ai
./run_frontend.sh
```

This will:
1. Check Python installation
2. Create a virtual environment
3. Install all dependencies
4. Start the web server at http://localhost:5000

### Option 2: Docker Compose (Recommended)

```bash
git clone https://github.com/groxaxo/autotune-ai.git
cd autotune-ai
docker-compose up autotune-ai
```

Access the web interface at: http://localhost:5000

## Docker Installation

### Prerequisites

1. **Install Docker**: https://docs.docker.com/get-docker/
2. **Install Docker Compose**: Usually included with Docker Desktop
3. **Install NVIDIA Container Toolkit** (for GPU support):

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu24.04 nvidia-smi
```

### Building the Image

```bash
# Clone repository
git clone https://github.com/groxaxo/autotune-ai.git
cd autotune-ai

# Build the image
docker build -t autotune-ai:latest -f docker/Dockerfile .
```

### Running with Docker Compose

**Web Interface:**
```bash
# Start in foreground
docker-compose up autotune-ai

# Start in background
docker-compose up -d autotune-ai

# View logs
docker-compose logs -f autotune-ai

# Stop service
docker-compose down
```

**Batch Processing:**
```bash
# Run batch processing
docker-compose run autotune-ai-batch snakemake -s Snakefile -j 4
```

### Running Manually with Docker

**Interactive Container:**
```bash
docker run --gpus all -it -v $(pwd):/work autotune-ai:latest
```

**Web Interface:**
```bash
docker run --gpus all -p 5000:5000 -it autotune-ai:latest python frontend/app.py
```

**CPU-Only (No GPU):**
```bash
docker run -p 5000:5000 -it autotune-ai:latest python frontend/app.py
```

## Local Installation

### Step 1: Install System Dependencies

**Ubuntu 24.04 / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3-pip \
    python3-venv \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential
```

**Fedora / RHEL:**
```bash
sudo dnf install -y \
    python3.12 \
    python3-pip \
    python3-devel \
    ffmpeg \
    libsndfile \
    libsndfile-devel \
    gcc \
    gcc-c++
```

### Step 2: Clone Repository

```bash
git clone https://github.com/groxaxo/autotune-ai.git
cd autotune-ai
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows
```

### Step 4: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 5: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Core audio processing (librosa, soundfile, scipy, numpy)
- Source separation (demucs)
- Pitch detection (crepe, pyworld)
- Deep learning (torch, torchaudio)
- Web frontend (flask, werkzeug)
- All other dependencies

### Step 6: Verify Installation

```bash
# Check Python packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import librosa; print('librosa:', librosa.__version__)"
python -c "import crepe; print('CREPE: OK')"
python -c "import pyworld; print('PyWorld: OK')"
python -c "import flask; print('Flask:', flask.__version__)"

# Check CUDA (if GPU available)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Run tests
pytest tests/ -v
```

## GPU Setup

### NVIDIA GPU Configuration

1. **Check GPU Compatibility:**
```bash
# List NVIDIA GPUs
nvidia-smi

# Check CUDA capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

Required: Compute Capability 6.0 or higher

2. **Install NVIDIA Drivers:**
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall
sudo reboot

# Check driver
nvidia-smi
```

3. **Install CUDA Toolkit:**

Download from: https://developer.nvidia.com/cuda-downloads

```bash
# Ubuntu 24.04 example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

4. **Install cuDNN (Optional but Recommended):**

Download from: https://developer.nvidia.com/cudnn

```bash
# Extract and copy files
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

5. **Verify PyTorch GPU Support:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"
python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### CPU-Only Setup

If you don't have an NVIDIA GPU, the system will automatically use CPU:

```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

# Run pipeline
python scripts/run_pipeline.py --input audio.wav --output corrected.wav
```

Performance tips for CPU:
- Use `--pitch_method librosa` (faster than CREPE)
- Use `--vocoder_method psola` (faster than WORLD)
- Reduce sample rate to 22050 Hz
- Process shorter audio segments

## Verification

### Test 1: Quick Test Script

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test
python examples/quick_test.py
```

This generates synthetic audio and tests the pipeline.

### Test 2: Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_utils.py -v
pytest tests/test_pitch.py -v
pytest tests/test_integration.py -v
```

### Test 3: Web Interface

```bash
# Start web server
cd frontend
python app.py

# Or use the quick start script
./run_frontend.sh
```

Access http://localhost:5000 and upload a test audio file.

### Test 4: Command Line Processing

```bash
# Create test audio (use any WAV/MP3 file)
# Process it
python scripts/run_pipeline.py \
    --input test_audio.wav \
    --output corrected.wav \
    --mode fast \
    --root_midi 60 \
    --scale major
```

## Troubleshooting

### Installation Issues

**Problem: `pip install` fails with compilation errors**

Solution:
```bash
# Install build dependencies
sudo apt-get install -y python3-dev build-essential

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Try again
pip install -r requirements.txt
```

**Problem: FFmpeg not found**

Solution:
```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg libsndfile1

# Verify
ffmpeg -version
```

**Problem: CUDA not detected**

Solution:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu122
```

**Problem: Out of memory during installation**

Solution:
```bash
# Increase swap space temporarily
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install dependencies
pip install -r requirements.txt

# Remove swap if needed
sudo swapoff /swapfile
sudo rm /swapfile
```

### Runtime Issues

**Problem: CUDA out of memory**

Solution:
```bash
# Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
```

**Problem: Slow processing**

Solution:
- Use `--pitch_method librosa` instead of `crepe`
- Use `--vocoder_method psola` instead of `world`
- Reduce sample rate: `--sr 22050`
- Ensure GPU is being used (check `nvidia-smi` during processing)

**Problem: Port 5000 already in use**

Solution:
```bash
# Change port
export PORT=8080
python frontend/app.py

# Or kill process using port 5000
sudo lsof -ti:5000 | xargs kill -9
```

**Problem: Permission denied on run_frontend.sh**

Solution:
```bash
chmod +x run_frontend.sh
./run_frontend.sh
```

### Docker Issues

**Problem: Docker can't access GPU**

Solution:
```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu24.04 nvidia-smi
```

**Problem: Docker build fails**

Solution:
```bash
# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t autotune-ai:latest -f docker/Dockerfile .
```

## Next Steps

After successful installation:

1. **Web Interface**: Access http://localhost:5000 and start processing audio
2. **Command Line**: See [README.md](README.md) for CLI usage examples
3. **Batch Processing**: See [USAGE.md](USAGE.md) for Snakemake workflows
4. **ML Training**: See model training documentation in [README.md](README.md)

## Getting Help

- **Documentation**: [README.md](README.md), [USAGE.md](USAGE.md)
- **GitHub Issues**: https://github.com/groxaxo/autotune-ai/issues
- **Discussions**: https://github.com/groxaxo/autotune-ai/discussions

## License

See [LICENSE](LICENSE) file for details.
