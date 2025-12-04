# Autotune-AI Quick Reference Guide

Quick commands and tips for common tasks.

## 🚀 Quick Start (Choose One)

```bash
# Option 1: Easiest - Quick Start Script
./run_frontend.sh
# Opens web interface at http://localhost:5000

# Option 2: Docker Compose (Recommended)
docker-compose up autotune-ai
# Opens web interface at http://localhost:5000

# Option 3: Local Installation
pip install -r requirements.txt
cd frontend && python app.py
# Opens web interface at http://localhost:5000
```

## 🌐 Web Interface

### Starting the Server
```bash
./run_frontend.sh                    # Easiest way
# OR
cd frontend && python app.py         # Manual start
# OR
docker-compose up autotune-ai        # Docker
```

### Accessing
- URL: http://localhost:5000
- Features: Upload, configure, process, download

## 💻 Command Line Usage

### Basic Processing
```bash
# Process mixed audio file
python scripts/run_pipeline.py \
    --input song.wav \
    --output corrected.wav \
    --mode fast

# Process with pre-separated stems
python scripts/run_pipeline.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output corrected.wav \
    --mode fast
```

### Custom Settings
```bash
python scripts/run_pipeline.py \
    --input song.wav \
    --output corrected.wav \
    --root_midi 62 \           # D4 key
    --scale minor \             # Minor scale
    --vibrato_preserve 0.3 \    # 30% vibrato
    --pitch_method crepe \      # CREPE for accuracy
    --vocoder_method world      # WORLD for quality
```

### Fast Processing (CPU)
```bash
python scripts/run_pipeline.py \
    --input song.wav \
    --output corrected.wav \
    --pitch_method librosa \    # Faster
    --vocoder_method psola \    # Faster
    --sr 22050                  # Lower sample rate
```

## 📦 Batch Processing

### Using Snakemake
```bash
# 1. Place audio files in data/input/
mkdir -p data/input
cp *.wav data/input/

# 2. Edit config
nano configs/config.yaml
# Add track names without .wav extension

# 3. Run batch processing
snakemake -s Snakefile -j 4  # 4 parallel jobs

# 4. Find results in results/
ls results/
```

## 🐳 Docker Commands

### Build
```bash
docker build -t autotune-ai:latest -f docker/Dockerfile .
```

### Run Web Interface
```bash
# With GPU
docker run --gpus all -p 5000:5000 -it autotune-ai:latest python frontend/app.py

# CPU only
docker run -p 5000:5000 -it autotune-ai:latest python frontend/app.py
```

### Interactive Container
```bash
docker run --gpus all -it -v $(pwd):/work autotune-ai:latest
```

### Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f autotune-ai

# Stop services
docker-compose down

# Batch processing
docker-compose run autotune-ai-batch snakemake -s Snakefile -j 4
```

## 🔧 Common Settings

### MIDI Notes
```
60 = C4 (Middle C)
62 = D4
64 = E4
65 = F4
67 = G4
69 = A4 (440 Hz)
```

### Scales
- `major` - Major scale (happy sound)
- `minor` - Minor scale (sad/emotional)
- `chromatic` - All 12 notes (minimal correction)

### Vibrato Preservation
- `0.0` - Full correction (robotic)
- `0.25` - Balanced (recommended)
- `0.5` - Natural sound
- `1.0` - No correction

### Methods
**Pitch Detection:**
- `crepe` - Accurate, slower (GPU recommended)
- `librosa` - Fast, good enough (CPU friendly)

**Vocoder:**
- `world` - High quality (recommended)
- `psola` - Fast, good quality

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_utils.py -v
pytest tests/test_pitch.py -v
pytest tests/test_integration.py -v

# With coverage
pytest tests/ -v --cov=scripts --cov=models

# Quick validation
python examples/quick_test.py
```

## 🔍 Debugging

### Check GPU
```bash
nvidia-smi                                    # Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Verbose Logging
```bash
export LOG_LEVEL=DEBUG
python scripts/run_pipeline.py ...
```

### Force CPU
```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/run_pipeline.py ...
```

### Profile Performance
```bash
time python scripts/run_pipeline.py ...
```

## 📝 Environment Variables

```bash
# Web Server
export PORT=5000              # Server port
export DEBUG=true             # Enable debug mode
export SECRET_KEY=xxx         # Flask secret key

# Processing
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
export CUDA_VISIBLE_DEVICES=""  # Force CPU

# Memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🎨 Web API

### Upload and Process
```bash
curl -X POST http://localhost:5000/upload \
  -F "audioFile=@song.wav" \
  -F "mode=fast" \
  -F "rootNote=60" \
  -F "scale=major"
# Returns: {"job_id": "xxx", "status": "queued"}
```

### Check Status
```bash
curl http://localhost:5000/status/{job_id}
# Returns: {"status": "processing", "progress": 50, ...}
```

### Download Result
```bash
curl -O http://localhost:5000/download/{job_id}
```

### Health Check
```bash
curl http://localhost:5000/health
```

## 📚 Documentation Quick Links

- **Installation**: [INSTALLATION.md](INSTALLATION.md)
- **Usage Guide**: [USAGE.md](USAGE.md)
- **Web Interface**: [frontend/README.md](frontend/README.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Main README**: [README.md](README.md)

## 🆘 Troubleshooting

### Issue: Port already in use
```bash
# Change port
export PORT=8080
python frontend/app.py

# Or kill process
sudo lsof -ti:5000 | xargs kill -9
```

### Issue: CUDA out of memory
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

### Issue: Slow processing
```bash
# Use faster methods
python scripts/run_pipeline.py \
    --pitch_method librosa \
    --vocoder_method psola \
    --sr 22050
```

### Issue: Dependencies not found
```bash
# Reinstall
pip install --force-reinstall -r requirements.txt

# Check installation
python -c "import flask, torch, librosa, crepe, pyworld"
```

## 💡 Tips & Tricks

1. **Web interface is easiest** - Start here if you're new
2. **Use GPU for speed** - 5-10x faster than CPU
3. **Test with short clips** - Verify settings before processing full songs
4. **Save your settings** - Use config.yaml for batch processing
5. **Check examples/** - Contains test scripts and sample workflows

## 🎵 Common Workflows

### Workflow 1: Correct a Single Song
1. Start web interface: `./run_frontend.sh`
2. Upload mixed audio file
3. Select key and scale
4. Click "Process Audio"
5. Download result

### Workflow 2: Batch Process Album
1. Place songs in `data/input/`
2. Edit `configs/config.yaml`
3. Run: `snakemake -s Snakefile -j 4`
4. Find results in `results/`

### Workflow 3: Fine-tune Settings
1. Process short clip with different settings
2. Listen to results
3. Adjust vibrato and scale settings
4. Process full song with best settings

---

For more details, see the full documentation in [README.md](README.md).
