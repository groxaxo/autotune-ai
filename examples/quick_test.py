#!/usr/bin/env python
"""Quick test script to verify the pipeline is working.

This script generates synthetic audio and runs it through the pipeline
to ensure all components are functioning correctly.
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from utils import write_wav, save_npz
from infer_target_pitch import midi_from_hz, hz_from_midi, heuristic_map


def generate_detuned_vocal(duration=3.0, sr=44100, base_note_midi=60):
    """
    Generate a synthetic vocal with intentional pitch variations.
    
    Simulates a slightly off-key singer by creating notes with small
    pitch deviations from the correct scale.
    """
    print(f'Generating {duration}s synthetic vocal...')
    
    # Create a simple melody (C major scale notes)
    notes = [60, 64, 67, 72, 67, 64, 60]  # C, E, G, C5, G, E, C
    note_duration = duration / len(notes)
    
    t = np.linspace(0, duration, int(sr * duration))
    y = np.zeros_like(t)
    
    # Generate each note with slight detuning
    for i, note_midi in enumerate(notes):
        start_idx = int(i * note_duration * sr)
        end_idx = int((i + 1) * note_duration * sr)
        
        if end_idx > len(t):
            end_idx = len(t)
        
        # Add random detuning (-30 to +30 cents)
        detune_cents = np.random.uniform(-30, 30)
        detune_midi = detune_cents / 100.0
        detuned_midi = note_midi + detune_midi
        
        # Convert to Hz
        freq = hz_from_midi(detuned_midi)
        
        # Generate sine wave for this note
        note_t = t[start_idx:end_idx] - t[start_idx]
        
        # Add vibrato (5-6 Hz)
        vibrato_freq = 5.5
        vibrato_depth = 10  # cents
        vibrato_cents = vibrato_depth * np.sin(2 * np.pi * vibrato_freq * note_t)
        freq_with_vibrato = hz_from_midi(detuned_midi + vibrato_cents / 100.0)
        
        # Generate waveform with time-varying frequency
        phase = np.cumsum(2 * np.pi * freq_with_vibrato / sr)
        y[start_idx:end_idx] = np.sin(phase)
        
        # Apply envelope (fade in/out)
        envelope = np.ones_like(note_t)
        fade_len = int(0.05 * sr)  # 50ms fade
        if len(envelope) > 2 * fade_len:
            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        
        y[start_idx:end_idx] *= envelope
    
    # Normalize
    y = y * 0.5 / (np.abs(y).max() + 1e-8)
    
    print(f'Generated vocal with {len(notes)} notes')
    return y, sr


def generate_backing(duration=3.0, sr=44100):
    """Generate a simple backing track (pad sound)."""
    print(f'Generating {duration}s backing track...')
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simple chord: C major (C, E, G)
    frequencies = [hz_from_midi(48), hz_from_midi(52), hz_from_midi(55)]  # C3, E3, G3
    
    y = np.zeros_like(t)
    for freq in frequencies:
        y += np.sin(2 * np.pi * freq * t) * 0.15
    
    # Add some variation
    y += np.sin(2 * np.pi * hz_from_midi(60) * t) * 0.1
    
    # Normalize
    y = y * 0.3 / (np.abs(y).max() + 1e-8)
    
    return y, sr


def extract_f0_simple(y, sr, hop_length=512):
    """
    Simple F0 extraction using zero-crossing rate.
    This is a placeholder for testing without CREPE.
    """
    print('Extracting F0 (simple method)...')
    
    n_frames = len(y) // hop_length
    times = np.arange(n_frames) * hop_length / sr
    f0 = np.zeros(n_frames)
    voiced_prob = np.ones(n_frames) * 0.8  # Assume mostly voiced
    
    # Simple estimation: assume notes are being generated
    # In real usage, this would be replaced by CREPE or librosa
    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = y[start:end]
        
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(frame)) != 0)
        if zero_crossings > 0:
            # Estimate frequency from zero crossings
            f0[i] = zero_crossings * sr / (2 * hop_length)
            if f0[i] < 80 or f0[i] > 1000:
                f0[i] = 0  # Out of vocal range
                voiced_prob[i] = 0.0
        else:
            voiced_prob[i] = 0.0
    
    return times, f0, voiced_prob


def main():
    print('=' * 60)
    print('Autotune-AI Quick Test')
    print('=' * 60)
    
    # Setup output directories
    output_dir = Path(__file__).parent / 'output'
    work_dir = output_dir / 'work'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test audio
    print('\n1. Generating synthetic audio...')
    vocal, sr = generate_detuned_vocal(duration=3.0, sr=22050)  # Lower SR for faster processing
    backing, _ = generate_backing(duration=3.0, sr=22050)
    
    # Save generated audio
    vocal_path = output_dir / 'test_vocal.wav'
    backing_path = output_dir / 'test_backing.wav'
    
    write_wav(vocal_path, vocal, sr)
    write_wav(backing_path, backing, sr)
    print(f'Saved: {vocal_path}')
    print(f'Saved: {backing_path}')
    
    # Extract F0
    print('\n2. Extracting pitch...')
    times, f0_hz, voiced_prob = extract_f0_simple(vocal, sr)
    
    f0_path = output_dir / 'f0_extracted.npz'
    save_npz(f0_path, times=times, f0_hz=f0_hz, voiced_prob=voiced_prob)
    print(f'Saved: {f0_path}')
    
    # Infer target pitch
    print('\n3. Inferring target pitch (C major scale)...')
    voiced = voiced_prob > 0.5
    target_f0 = heuristic_map(
        f0_hz, voiced,
        root_midi=60,  # C4
        scale='major',
        vibrato_preserve=0.25
    )
    
    target_path = output_dir / 'f0_target.npz'
    save_npz(target_path, times=times, f0_hz=target_f0, voiced_prob=voiced_prob)
    print(f'Saved: {target_path}')
    
    # Summary
    print('\n' + '=' * 60)
    print('Test completed successfully!')
    print('=' * 60)
    print('\nGenerated files:')
    print(f'  - {vocal_path}')
    print(f'  - {backing_path}')
    print(f'  - {f0_path}')
    print(f'  - {target_path}')
    print('\nNext steps:')
    print('  1. Listen to the generated audio files')
    print('  2. Run the full pipeline with:')
    print(f'     python scripts/run_pipeline.py \\')
    print(f'       --vocal {vocal_path} \\')
    print(f'       --backing {backing_path} \\')
    print(f'       --output {output_dir}/corrected.wav \\')
    print(f'       --mode fast')
    print('\nNote: This quick test uses simplified F0 extraction.')
    print('For real audio, use the full pipeline with CREPE or librosa.')


if __name__ == '__main__':
    main()
