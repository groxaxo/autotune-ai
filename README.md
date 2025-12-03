# Autotune-AI

An advanced AI-powered audio pitch correction pipeline for Ubuntu 24.04 with NVIDIA GPU support. This project implements a complete autotune solution using state-of-the-art audio processing libraries and deep learning models.

## Features

- **Source Separation**: Separate vocals from backing tracks using Demucs
- **Pitch Detection**: Extract pitch using CREPE (deep learning) or librosa pyin
- **Intelligent Pitch Correction**: Snap pitches to musical scales while preserving vibrato
- **High-Quality Vocoding**: WORLD vocoder for pitch manipulation
- **Post-Processing**: Loudness normalization, de-essing, and professional mixing
- **ML Model Support**: Optional deep learning model for advanced pitch prediction
- **GPU Acceleration**: Full CUDA support for fast processing

## System Requirements

- Ubuntu 24.04 LTS
- NVIDIA GPU with CUDA support (recommended)
- Python 3.12+
- FFmpeg
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space

## Quick Start

### Installation

#### Option 1: Docker (Recommended)

```bash
# Build Docker image
docker build -t autotune-ai:latest -f docker/Dockerfile .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/work autotune-ai:latest

# Inside container, verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Option 2: Local Installation

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.12 python3-pip ffmpeg libsndfile1

# Install Python dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Process a Single Audio File

```bash
# Using pre-separated vocals and backing
python scripts/run_pipeline.py \
    --vocal path/to/vocal.wav \
    --backing path/to/backing.wav \
    --output results/corrected.wav \
    --mode fast

# With source separation
python scripts/run_pipeline.py \
    --input path/to/mixed.wav \
    --output results/corrected.wav \
    --mode fast
```

#### Using Snakemake for Batch Processing

```bash
# 1. Place input audio files in data/input/ (e.g., data/input/song1.wav)

# 2. Edit configs/config.yaml to list your tracks:
#    tracks:
#      - song1
#      - song2

# 3. Run the pipeline
snakemake -s Snakefile -j 4

# Results will be in results/ directory
```

## Pipeline Overview

The pipeline consists of six main stages:

1. **Separation** (`scripts/separate.py`)
   - Separates vocals from backing track using Demucs
   - Outputs: `vocal.wav`, `instr.wav`

2. **Preprocessing** (`scripts/preprocess.py`)
   - Optional noise reduction
   - Voice activity detection
   - Time alignment with backing track

3. **Pitch Extraction** (`scripts/extract_pitch.py`)
   - Extracts F0 contour using CREPE or librosa
   - Outputs: NPZ file with times, F0, and voicing probability

4. **Target Pitch Inference** (`scripts/infer_target_pitch.py`)
   - Heuristic mode: Snap to musical scale
   - Model mode: Use trained ML model (optional)

5. **Pitch Correction** (`scripts/correct_pitch.py`)
   - Apply correction using WORLD vocoder
   - Preserves voice timbre and expression

6. **Post-Processing** (`scripts/postprocess.py`)
   - Loudness normalization (LUFS)
   - De-essing
   - Mix with backing track

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Musical parameters
root_midi: 60        # Root note (60 = C4)
scale: major         # major, minor, or chromatic
vibrato_preserve: 0.25  # 0-1, amount of vibrato to keep

# Processing options
denoise: true        # Apply noise reduction
pitch_method: crepe  # crepe or librosa
vocoder_method: world  # world or psola
```

## Advanced Usage

### Custom Pitch Correction

```bash
# Fine-tune correction strength
python scripts/run_pipeline.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output corrected.wav \
    --vibrato_preserve 0.5 \
    --root_midi 65 \
    --scale minor
```

### Preprocessing Options

```bash
# With denoising and alignment
python scripts/run_pipeline.py \
    --input mixed.wav \
    --output corrected.wav \
    --denoise \
    --align
```

### GPU Selection

```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

## Machine Learning Model (Optional)

### Training a Custom Model

```bash
# 1. Prepare training data in data/train/
#    Structure:
#    data/train/
#      pair_0001/
#        detuned_vocal.wav
#        target.npz
#        backing.wav (optional)

# 2. Train model
python models/pitch_predictor/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 100 \
    --batch_size 16 \
    --output_dir models/checkpoints

# 3. Use trained model
python scripts/run_pipeline.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output corrected.wav \
    --mode model \
    --model_ckpt models/checkpoints/best.pt
```

## Script Reference

### Individual Scripts

All scripts can be run independently:

```bash
# Separate audio
python scripts/separate.py -i input.wav -d output_dir/

# Extract pitch
python scripts/extract_pitch.py -i vocal.wav -o f0.npz --method crepe

# Infer target pitch
python scripts/infer_target_pitch.py --f0_npz f0.npz -o target.npz \
    --root_midi 60 --scale major

# Correct pitch
python scripts/correct_pitch.py -v vocal.wav --target_npz target.npz -o corrected.wav

# Mix final output
python scripts/postprocess.py -v corrected.wav -b backing.wav -o final.wav
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller batch sizes
python models/pitch_predictor/train.py --batch_size 8

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

### Demucs Separation Fails

```bash
# Try a different model
python scripts/separate.py --model htdemucs_ft

# Or use pre-separated vocals
python scripts/run_pipeline.py --vocal vocal.wav --backing backing.wav ...
```

### Poor Pitch Correction Quality

```bash
# Increase vibrato preservation
--vibrato_preserve 0.5

# Use chromatic scale for no quantization
--scale chromatic

# Try PSOLA instead of WORLD
--vocoder_method psola
```

## Performance

Typical processing times on RTX 3080:
- Separation (3-minute song): ~30 seconds
- Pitch extraction (CREPE): ~45 seconds
- Pitch correction: ~10 seconds
- Total pipeline: ~2 minutes

CPU-only (Intel i7):
- Total pipeline: ~10-15 minutes

## Project Structure

```
autotune-ai/
├── configs/           # Configuration files
├── data/             # Data directories
│   ├── input/        # Input audio files
│   ├── separation/   # Separated stems
│   ├── preprocessed/ # Preprocessed audio
│   ├── f0/          # Extracted F0 contours
│   ├── target/      # Target F0 contours
│   └── corrected/   # Corrected vocals
├── docker/          # Docker configuration
├── models/          # ML models
│   ├── pitch_predictor/  # Pitch prediction model
│   └── vocoder/     # Vocoder utilities
├── results/         # Final outputs
├── scripts/         # Processing scripts
└── tests/          # Unit tests

```

## Dependencies

Core libraries:
- **Demucs**: Source separation
- **CREPE**: Deep learning pitch detection
- **pyworld**: Pitch manipulation vocoder
- **librosa**: Audio analysis
- **PyTorch**: Deep learning framework
- **noisereduce**: Noise reduction
- **pyloudnorm**: Loudness normalization

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Areas for improvement:
- Add more musical scales and modes
- Improve ML model architecture
- Add real-time processing support
- Implement HiFi-GAN vocoder integration
- Add GUI interface

## License

This project is for educational and research purposes. Please ensure you have appropriate rights to process any audio files.

## Acknowledgments

- Demucs by Facebook Research
- CREPE by Jong Wook Kim et al.
- WORLD vocoder by Masanori Morise
- Librosa by Brian McFee et al.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Open a new issue with:
   - System information
   - Error messages
   - Steps to reproduce

## Roadmap

- [x] Core pipeline implementation
- [x] GPU acceleration support
- [x] Docker containerization
- [x] Batch processing with Snakemake
- [ ] ML model training dataset generation
- [ ] HiFi-GAN vocoder integration
- [ ] Real-time processing mode
- [ ] Web interface
- [ ] VST plugin

---

Built with ❤️ for audio engineers and musicians
