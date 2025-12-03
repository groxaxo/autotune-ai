# Examples

This directory contains example files and scripts for testing the autotune-ai pipeline.

## Quick Test Script

The `quick_test.py` script generates synthetic audio and runs it through the pipeline to verify everything is working.

```bash
python examples/quick_test.py
```

This will:
1. Generate a synthetic vocal recording (sine wave with slight pitch variations)
2. Run it through the fast pipeline
3. Save the corrected output to `examples/output/test_corrected.wav`

## Example Audio Files

Due to size constraints, example audio files are not included in the repository. You can:

1. **Use your own audio files**: Place them in `data/input/` directory
2. **Generate synthetic examples**: Run the quick test script
3. **Download test datasets**: Check the main README for dataset links

## Expected Output Structure

After running the pipeline on an example, you'll get:

```
examples/output/
├── test_corrected.wav      # Final corrected mix
└── work/                   # Intermediate files
    ├── separation/
    │   ├── vocal.wav
    │   └── instr.wav
    ├── preprocessed/
    │   └── vocal_pre.wav
    ├── f0/
    │   └── extracted.npz
    ├── target/
    │   └── target.npz
    └── corrected/
        └── vocal_corrected.wav
```

## Testing Different Parameters

You can test different pitch correction settings:

```bash
# Minimal correction (preserve most of original pitch)
python scripts/run_pipeline.py \
    --vocal examples/test_vocal.wav \
    --backing examples/test_backing.wav \
    --output examples/output/minimal.wav \
    --vibrato_preserve 0.8 \
    --scale chromatic

# Strong correction (auto-tune effect)
python scripts/run_pipeline.py \
    --vocal examples/test_vocal.wav \
    --backing examples/test_backing.wav \
    --output examples/output/strong.wav \
    --vibrato_preserve 0.0 \
    --scale major
```

## Troubleshooting

If the quick test fails:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify Python version: `python --version` (should be 3.12+)
3. Check for error messages in the console output
