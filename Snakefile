"""
Snakemake workflow for autotune-ai pipeline.
Run with: snakemake -s Snakefile -j <cores>
"""

import yaml
from pathlib import Path

# Load configuration
configfile: "configs/config.yaml"

# Get list of tracks from config
TRACKS = config.get("tracks", [])

# Set global parameters
SR = config.get("sample_rate", 44100)
MODE = config.get("mode", "fast")
SEPARATION_MODEL = config.get("separation_model", "htdemucs")
PITCH_METHOD = config.get("pitch_method", "crepe")
ROOT_MIDI = config.get("root_midi", 60)
SCALE = config.get("scale", "major")
VOCODER = config.get("vocoder_method", "world")

# Rule to process all tracks
rule all:
    input:
        expand("results/{track}.wav", track=TRACKS)

# Rule: Separate audio into stems
rule separate:
    input:
        audio="data/input/{track}.wav"
    output:
        vocal="data/separation/{track}/vocal.wav",
        instr="data/separation/{track}/instr.wav"
    params:
        out_dir="data/separation/{track}",
        model=SEPARATION_MODEL
    shell:
        """
        python scripts/separate.py \
            --input {input.audio} \
            --out_dir {params.out_dir} \
            --model {params.model}
        """

# Rule: Preprocess vocal
rule preprocess:
    input:
        vocal="data/separation/{track}/vocal.wav",
        backing="data/separation/{track}/instr.wav"
    output:
        "data/preprocessed/{track}/vocal_pre.wav"
    params:
        sr=SR
    shell:
        """
        python scripts/preprocess.py \
            --vocal {input.vocal} \
            --output {output} \
            --sr {params.sr} \
            --denoise
        """

# Rule: Extract F0
rule extract_f0:
    input:
        "data/preprocessed/{track}/vocal_pre.wav"
    output:
        "data/f0/{track}.npz"
    params:
        sr=SR,
        method=PITCH_METHOD
    shell:
        """
        python scripts/extract_pitch.py \
            --input {input} \
            --output {output} \
            --sr {params.sr} \
            --method {params.method}
        """

# Rule: Infer target pitch
rule infer_target:
    input:
        "data/f0/{track}.npz"
    output:
        "data/target/{track}.npz"
    params:
        mode=MODE,
        root_midi=ROOT_MIDI,
        scale=SCALE
    shell:
        """
        python scripts/infer_target_pitch.py \
            --f0_npz {input} \
            --output {output} \
            --mode {params.mode} \
            --root_midi {params.root_midi} \
            --scale {params.scale}
        """

# Rule: Correct pitch
rule correct_pitch:
    input:
        vocal="data/preprocessed/{track}/vocal_pre.wav",
        target="data/target/{track}.npz"
    output:
        "data/corrected/{track}/vocal_corrected.wav"
    params:
        sr=SR,
        method=VOCODER
    shell:
        """
        python scripts/correct_pitch.py \
            --vocal {input.vocal} \
            --target_npz {input.target} \
            --output {output} \
            --sr {params.sr} \
            --method {params.method}
        """

# Rule: Post-process and mix
rule postprocess:
    input:
        vocal="data/corrected/{track}/vocal_corrected.wav",
        backing="data/separation/{track}/instr.wav"
    output:
        "results/{track}.wav"
    params:
        sr=SR
    shell:
        """
        python scripts/postprocess.py \
            --vocal {input.vocal} \
            --backing {input.backing} \
            --output {output} \
            --sr {params.sr} \
            --deess
        """

# Clean intermediate files
rule clean:
    shell:
        """
        rm -rf data/separation data/preprocessed data/f0 data/target data/corrected
        """

# Clean everything including results
rule clean_all:
    shell:
        """
        rm -rf data/separation data/preprocessed data/f0 data/target data/corrected results
        """
