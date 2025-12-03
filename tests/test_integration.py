"""Integration tests for the pipeline."""
import pytest
import numpy as np
import tempfile
import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from utils import write_wav, read_wav, save_npz


def generate_test_audio(duration=1.0, sr=44100, freq=440.0):
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * freq * t) * 0.5
    return y, sr


def test_generate_test_audio():
    """Test audio generation utility."""
    y, sr = generate_test_audio(duration=1.0, sr=44100)
    assert len(y) == 44100
    assert sr == 44100
    assert np.abs(y).max() <= 0.5


def test_audio_round_trip():
    """Test writing and reading audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate test audio
        y_orig, sr = generate_test_audio(duration=0.5, sr=22050)
        
        # Write
        wav_path = Path(tmpdir) / 'test.wav'
        write_wav(wav_path, y_orig, sr)
        
        # Read
        y_loaded, sr_loaded = read_wav(wav_path, sr=sr)
        
        # Verify
        assert sr_loaded == sr
        assert len(y_loaded) > 0


def test_npz_workflow():
    """Test NPZ file workflow for F0 data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / 'f0_data.npz'
        
        # Create mock F0 data
        n_frames = 100
        times = np.linspace(0, 2.0, n_frames)
        f0_hz = np.random.rand(n_frames) * 100 + 200  # 200-300 Hz
        voiced_prob = np.random.rand(n_frames)
        
        # Save
        save_npz(npz_path, times=times, f0_hz=f0_hz, voiced_prob=voiced_prob)
        
        # Verify file exists
        assert npz_path.exists()


def test_pitch_conversion_utilities():
    """Test MIDI/Hz conversion utilities."""
    from infer_target_pitch import midi_from_hz, hz_from_midi
    
    # Test known values
    assert abs(midi_from_hz(440.0) - 69.0) < 0.01  # A4
    assert abs(hz_from_midi(60.0) - 261.63) < 1.0  # C4
    
    # Test round-trip
    for freq in [100, 200, 300, 400, 500]:
        midi = midi_from_hz(freq)
        freq_back = hz_from_midi(midi)
        assert abs(freq_back - freq) < 0.1


def test_scale_definitions():
    """Test musical scale definitions."""
    from infer_target_pitch import SCALES
    
    # Verify scales exist
    assert 'major' in SCALES
    assert 'minor' in SCALES
    assert 'chromatic' in SCALES
    
    # Verify scale properties
    assert len(SCALES['major']) == 7
    assert len(SCALES['minor']) == 7
    assert len(SCALES['chromatic']) == 12
    
    # Verify root note is in each scale
    assert 0 in SCALES['major']
    assert 0 in SCALES['minor']
    assert 0 in SCALES['chromatic']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
