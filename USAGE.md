# Autotune-AI Usage Guide

Detailed guide for using the autotune-ai pipeline.

## Table of Contents

1. [Quick Start Examples](#quick-start-examples)
2. [Individual Script Usage](#individual-script-usage)
3. [Configuration Guide](#configuration-guide)
4. [Batch Processing](#batch-processing)
5. [ML Model Training](#ml-model-training)
6. [Advanced Topics](#advanced-topics)

## Quick Start Examples

### Example 1: Basic Pitch Correction

Correct pitch of a vocal recording with backing track:

```bash
python scripts/run_pipeline.py \
    --vocal my_vocal.wav \
    --backing my_backing.wav \
    --output corrected.wav \
    --mode fast
```

### Example 2: Auto-Separate and Correct

Process a mixed audio file (automatically separates vocal and backing):

```bash
python scripts/run_pipeline.py \
    --input mixed_song.wav \
    --output corrected.wav \
    --mode fast
```

### Example 3: Custom Key and Scale

Correct to D minor with strong correction:

```bash
python scripts/run_pipeline.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output corrected.wav \
    --root_midi 62 \
    --scale minor \
    --vibrato_preserve 0.1
```

### Example 4: With Preprocessing

Apply denoising and alignment:

```bash
python scripts/run_pipeline.py \
    --input mixed.wav \
    --output corrected.wav \
    --denoise \
    --align
```

## Individual Script Usage

### 1. Source Separation (`separate.py`)

Extract vocal and instrumental stems from a mixed audio file.

```bash
python scripts/separate.py \
    --input song.wav \
    --out_dir output/stems/ \
    --model htdemucs
```

**Options:**
- `--input, -i`: Input audio file (required)
- `--out_dir, -d`: Output directory (required)
- `--model`: Demucs model (`htdemucs`, `htdemucs_ft`, `mdx_extra`)
- `--device`: Processing device (`cuda`, `cpu`)

**Output:**
- `output/stems/vocal.wav`: Isolated vocal track
- `output/stems/instr.wav`: Instrumental/backing track

### 2. Preprocessing (`preprocess.py`)

Clean and prepare vocal audio for pitch extraction.

```bash
python scripts/preprocess.py \
    --vocal raw_vocal.wav \
    --output clean_vocal.wav \
    --denoise \
    --vad \
    --sr 44100
```

**Options:**
- `--vocal, -v`: Input vocal file (required)
- `--output, -o`: Output file (required)
- `--backing, -b`: Backing track for alignment
- `--denoise`: Apply noise reduction
- `--vad`: Remove silence with voice activity detection
- `--align`: Align vocal to backing track
- `--sr`: Sample rate (default: 44100)
- `--prop_decrease`: Noise reduction strength 0-1 (default: 0.8)
- `--top_db`: VAD threshold in dB (default: 30)

**Example with alignment:**
```bash
python scripts/preprocess.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output aligned_vocal.wav \
    --denoise \
    --align
```

### 3. Pitch Extraction (`extract_pitch.py`)

Extract fundamental frequency (F0) from vocal audio.

```bash
python scripts/extract_pitch.py \
    --input vocal.wav \
    --output f0_data.npz \
    --method crepe \
    --sr 44100
```

**Options:**
- `--input, -i`: Input audio file (required)
- `--output, -o`: Output NPZ file (required)
- `--method`: Extraction method (`crepe`, `librosa`)
- `--sr`: Sample rate
- `--model_capacity`: CREPE model size (`tiny`, `small`, `medium`, `large`, `full`)
- `--device`: Processing device
- `--fmin`: Minimum frequency in Hz
- `--fmax`: Maximum frequency in Hz

**Output NPZ contains:**
- `times`: Time stamps (seconds)
- `f0_hz`: F0 values in Hz
- `voiced_prob`: Voicing probability (0-1)

**Method comparison:**
- **CREPE**: More accurate, slower, GPU-accelerated
- **librosa**: Faster, CPU-friendly, less accurate

### 4. Target Pitch Inference (`infer_target_pitch.py`)

Determine target pitches for correction.

```bash
python scripts/infer_target_pitch.py \
    --f0_npz extracted_f0.npz \
    --output target_f0.npz \
    --mode heuristic \
    --root_midi 60 \
    --scale major \
    --vibrato_preserve 0.25
```

**Options:**
- `--f0_npz`: Input F0 NPZ file (required)
- `--output, -o`: Output target NPZ file (required)
- `--mode`: `heuristic` or `model`
- `--root_midi`: Root note (MIDI number, 60=C4)
- `--scale`: Musical scale (`major`, `minor`, `chromatic`)
- `--vibrato_preserve`: Vibrato preservation 0-1
- `--snap_threshold`: Max snap distance in semitones
- `--model_ckpt`: Model checkpoint (for model mode)

**Note**: model mode currently falls back to fast heuristic mode — model inference is not yet implemented.

**MIDI Note Reference:**
```
C4 = 60  (Middle C)
D4 = 62
E4 = 64
F4 = 65
G4 = 67
A4 = 69  (440 Hz)
B4 = 71
C5 = 72
```

**Scale Options:**
- `major`: Natural major scale
- `minor`: Natural minor scale
- `chromatic`: All 12 semitones (minimal correction)

### 5. Pitch Correction (`correct_pitch.py`)

Apply pitch correction using vocoder.

```bash
python scripts/correct_pitch.py \
    --vocal preprocessed_vocal.wav \
    --target_npz target_f0.npz \
    --output corrected_vocal.wav \
    --method world \
    --sr 44100
```

**Options:**
- `--vocal, -v`: Input vocal file (required)
- `--target_npz`: Target F0 NPZ file (required)
- `--output, -o`: Output file (required)
- `--method`: Vocoder method (`world`, `psola`)
- `--sr`: Sample rate

**Method comparison:**
- **WORLD**: High quality, preserves timbre
- **PSOLA**: Faster, simpler, may have artifacts

### 6. Post-Processing (`postprocess.py`)

Finalize and mix corrected vocal with backing.

```bash
python scripts/postprocess.py \
    --vocal corrected_vocal.wav \
    --backing backing.wav \
    --output final_mix.wav \
    --deess \
    --vocal_gain_db 2.0 \
    --target_lufs -14.0
```

**Options:**
- `--vocal, -v`: Corrected vocal file (required)
- `--backing, -b`: Backing track file (required)
- `--output, -o`: Output file (required)
- `--vocal_gain_db`: Vocal level adjustment (default: 0)
- `--backing_gain_db`: Backing level adjustment (default: 0)
- `--target_lufs`: Target loudness in LUFS (default: -14)
- `--deess`: Apply de-essing
- `--deess_freq`: De-esser frequency in Hz (default: 5000)

## Configuration Guide

### Edit `configs/config.yaml`

```yaml
# List tracks to process
tracks:
  - song1
  - song2

# Global parameters
sample_rate: 44100
mode: fast  # or 'model'

# Pitch correction settings
root_midi: 60
scale: major
vibrato_preserve: 0.25

# Quality settings
pitch_method: crepe  # or 'librosa'
vocoder_method: world  # or 'psola'

# Preprocessing
denoise: true
vad: false
align: false
```

## Batch Processing

### Using Snakemake

1. **Prepare data structure:**
```
data/
  input/
    song1.wav
    song2.wav
    song3.wav
```

2. **Configure tracks in `configs/config.yaml`:**
```yaml
tracks:
  - song1
  - song2
  - song3
```

3. **Run pipeline:**
```bash
# Process all tracks with 4 parallel jobs
snakemake -s Snakefile -j 4

# Process specific track
snakemake -s Snakefile results/song1.wav

# Dry run (show what would be done)
snakemake -s Snakefile -n

# Clean intermediate files
snakemake -s Snakefile clean
```

### Processing Multiple Files with Script

```bash
#!/bin/bash
# process_batch.sh

for file in data/input/*.wav; do
    basename=$(basename "$file" .wav)
    echo "Processing $basename..."
    
    python scripts/run_pipeline.py \
        --input "$file" \
        --output "results/${basename}_corrected.wav" \
        --mode fast \
        --denoise
done
```

## ML Model Training

### Data Preparation

Create paired training data:

```
data/train/
  pair_0001/
    detuned_vocal.wav    # Vocal with pitch errors
    target.npz           # Ground truth F0
    backing.wav          # Optional backing track
  pair_0002/
    ...
```

### Generate Training Data

Create synthetic detuned vocals:

```python
import librosa
import numpy as np

# Load clean vocal
y, sr = librosa.load('clean_vocal.wav', sr=44100)

# Apply random pitch shift
n_steps = np.random.uniform(-2, 2)  # ±2 semitones
y_detuned = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# Save detuned version
librosa.output.write_wav('detuned_vocal.wav', y_detuned, sr)

# Extract and save ground truth F0
# ... (use extract_pitch.py on original)
```

### Train Model

```bash
python models/pitch_predictor/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --output_dir models/checkpoints
```

### Use Trained Model

```bash
python scripts/run_pipeline.py \
    --vocal vocal.wav \
    --backing backing.wav \
    --output corrected.wav \
    --mode model \
    --model_ckpt models/checkpoints/best.pt
```

## Advanced Topics

### Custom Vibrato Control

Preserve more vibrato (natural sound):
```bash
--vibrato_preserve 0.5  # 50% preservation
```

Less vibrato (robotic sound):
```bash
--vibrato_preserve 0.0  # Full correction
```

### Processing Different Vocal Ranges

**Male vocals (lower range):**
```bash
--root_midi 55  # G3
--fmin 80       # Lower minimum frequency
```

**Female vocals (higher range):**
```bash
--root_midi 67  # G4
--fmax 1000     # Higher maximum frequency
```

**Children's vocals:**
```bash
--root_midi 72  # C5
--fmin 150
--fmax 1500
```

### Extreme Pitch Correction

For heavily out-of-tune vocals:
```bash
--vibrato_preserve 0.0
--scale chromatic
--snap_threshold 2.0
```

### Subtle Enhancement

For already good vocals:
```bash
--vibrato_preserve 0.8
--snap_threshold 0.25
```

### GPU Memory Management

```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
```

### Real-time Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor processing
tail -f logs/pipeline.log
```

## Troubleshooting Tips

### Poor Separation Quality
- Try different Demucs models: `htdemucs_ft` or `mdx_extra`
- Ensure input is high quality (not compressed)
- Use longer audio segments (>30 seconds)

### Robotic Vocal Sound
- Increase `--vibrato_preserve` to 0.5 or higher
- Try `--vocoder_method psola`
- Use `--scale chromatic` for less quantization

### Artifacts in Output
- Enable `--deess` for sibilance
- Reduce `--vocal_gain_db`
- Use gentler `--snap_threshold`

### Slow Processing
- Use `--pitch_method librosa` instead of CREPE
- Reduce batch size in model training
- Process shorter segments

## Performance Benchmarks

Typical processing times (3-minute song):

| Stage | GPU (RTX 3080) | CPU (i7) |
|-------|----------------|----------|
| Separation | 30s | 5m |
| Preprocessing | 5s | 15s |
| Pitch Extract (CREPE) | 45s | 8m |
| Pitch Extract (librosa) | 10s | 30s |
| Correction | 10s | 30s |
| Post-process | 5s | 10s |
| **Total (CREPE)** | **~2m** | **~15m** |
| **Total (librosa)** | **~1m** | **~7m** |

---

For more information, see [README.md](README.md)
