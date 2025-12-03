"""Tests for utility functions."""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from utils import (
    ensure_dir, read_wav, write_wav, save_npz, load_npz,
    normalize_audio, safe_divide
)


def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / 'test' / 'nested' / 'dir'
        ensure_dir(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()


def test_safe_divide():
    """Test safe division function."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 0.0, 3.0])
    
    result = safe_divide(a, b, fill_value=0.0)
    
    assert result[0] == 0.5
    assert result[1] == 0.0  # Division by zero filled
    assert result[2] == 1.0


def test_normalize_audio():
    """Test audio normalization."""
    # Create test signal
    y = np.random.randn(1000) * 0.1
    
    # Normalize to -20 dB
    y_norm = normalize_audio(y, target_db=-20.0)
    
    # Check RMS is approximately correct
    rms = np.sqrt(np.mean(y_norm**2))
    rms_db = 20 * np.log10(rms)
    
    assert abs(rms_db - (-20.0)) < 1.0  # Within 1 dB


def test_npz_save_load():
    """Test NPZ file save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / 'test.npz'
        
        # Create test data
        times = np.linspace(0, 10, 100)
        f0 = np.random.rand(100) * 100 + 200
        voiced = np.random.rand(100)
        
        # Save
        save_npz(npz_path, times=times, f0_hz=f0, voiced_prob=voiced)
        
        # Load
        data = load_npz(npz_path)
        
        # Check
        assert 'times' in data
        assert 'f0_hz' in data
        assert 'voiced_prob' in data
        assert np.allclose(data['times'], times)
        assert np.allclose(data['f0_hz'], f0)
        assert np.allclose(data['voiced_prob'], voiced)


def test_audio_write_read():
    """Test audio write and read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / 'test.wav'
        
        # Create test audio
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
        
        # Write
        write_wav(wav_path, y, sr)
        
        # Read
        y_loaded, sr_loaded = read_wav(wav_path, sr=sr)
        
        # Check
        assert sr_loaded == sr
        assert len(y_loaded) > 0
        # Note: Some difference expected due to compression


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
