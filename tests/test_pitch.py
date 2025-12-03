"""Tests for pitch processing functions."""
import pytest
import numpy as np
import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from infer_target_pitch import (
    midi_from_hz, hz_from_midi, snap_to_scale,
    heuristic_map, SCALES
)


def test_midi_hz_conversion():
    """Test MIDI to Hz and Hz to MIDI conversion."""
    # A4 = 440 Hz = MIDI 69
    assert abs(midi_from_hz(440.0) - 69.0) < 0.01
    assert abs(hz_from_midi(69.0) - 440.0) < 0.01
    
    # C4 = ~261.63 Hz = MIDI 60
    assert abs(midi_from_hz(261.63) - 60.0) < 0.1
    assert abs(hz_from_midi(60.0) - 261.63) < 0.5
    
    # Test round-trip
    test_hz = 300.0
    midi = midi_from_hz(test_hz)
    hz = hz_from_midi(midi)
    assert abs(hz - test_hz) < 0.01


def test_snap_to_scale():
    """Test scale snapping function."""
    # Test C major scale (root = 60)
    root = 60
    
    # C (60) should stay C
    assert snap_to_scale(60.0, root, 'major') == 60.0
    
    # C# (61) should snap to C (60) or D (62), or stay if outside threshold
    snapped = snap_to_scale(61.0, root, 'major', snap_threshold=0.5)
    # With threshold 0.5, 61 is exactly 0.5 away from 60 and 1.0 away from 62
    # So it should snap to 60 or stay at 61 depending on implementation
    assert snapped in [60.0, 61.0, 62.0]
    
    # D (62) should stay D (in C major scale)
    assert snap_to_scale(62.0, root, 'major') == 62.0
    
    # Chromatic scale should keep everything
    assert snap_to_scale(61.0, root, 'chromatic') == 61.0


def test_heuristic_map():
    """Test heuristic pitch mapping."""
    # Create test F0 contour
    n_frames = 100
    
    # Generate slightly detuned notes
    # Note around C4 (261.63 Hz)
    f0 = np.ones(n_frames) * 265.0  # Slightly sharp
    voiced = np.ones(n_frames, dtype=bool)
    
    # Apply heuristic mapping
    target_f0 = heuristic_map(
        f0, voiced,
        root_midi=60,  # C4
        scale='major',
        vibrato_preserve=0.0,  # No vibrato for testing
        snap_threshold=0.5
    )
    
    # Check that output is in expected range
    assert len(target_f0) == len(f0)
    assert np.all(target_f0[voiced] > 0)
    
    # Check that correction was applied (should be closer to 261.63)
    assert np.mean(target_f0[voiced]) < 265.0


def test_scales():
    """Test scale definitions."""
    # Major scale should have 7 notes
    assert len(SCALES['major']) == 7
    
    # Minor scale should have 7 notes
    assert len(SCALES['minor']) == 7
    
    # Chromatic scale should have 12 notes
    assert len(SCALES['chromatic']) == 12
    
    # Major scale should contain specific intervals
    assert 0 in SCALES['major']  # Root
    assert 2 in SCALES['major']  # Major 2nd
    assert 4 in SCALES['major']  # Major 3rd
    assert 7 in SCALES['major']  # Perfect 5th


def test_voiced_handling():
    """Test handling of unvoiced frames."""
    n_frames = 100
    
    # Create F0 with some unvoiced frames
    f0 = np.random.rand(n_frames) * 100 + 200
    voiced = np.random.rand(n_frames) > 0.5  # 50% voiced
    
    # Apply mapping
    target_f0 = heuristic_map(f0, voiced, root_midi=60, scale='major')
    
    # Unvoiced frames should be 0
    assert np.all(target_f0[~voiced] == 0.0)
    
    # Voiced frames should be non-zero
    if np.any(voiced):
        assert np.all(target_f0[voiced] > 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
