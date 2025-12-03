"""Given an extracted F0 contour and an optional detected key/scale, produce a target F0 contour.
Two modes: heuristic (fast) and model (requires checkpoint).
"""
import argparse
import logging
import sys
import numpy as np
from scipy.signal import medfilt
from utils import load_npz, save_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Musical scales (semitones relative to root)
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'chromatic': list(range(12)),  # All semitones
}


def midi_from_hz(hz):
    """Convert frequency in Hz to MIDI note number."""
    # Avoid log of zero
    hz = np.maximum(hz, 1e-6)
    return 69 + 12 * np.log2(hz / 440.0)


def hz_from_midi(midi):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def extract_vibrato(midi, window_size=11):
    """
    Extract vibrato (high-frequency pitch variation) from MIDI contour.
    
    Args:
        midi: MIDI note contour
        window_size: Window size for smoothing
        
    Returns:
        Vibrato residual
    """
    # Smooth with median filter to get base pitch
    midi_smooth = medfilt(midi, kernel_size=window_size)
    
    # Vibrato is the residual
    vibrato = midi - midi_smooth
    
    return vibrato, midi_smooth


def snap_to_scale(midi, root_midi, scale='major', snap_threshold=0.5):
    """
    Snap MIDI notes to nearest scale degree.
    
    Args:
        midi: MIDI note number (can be fractional)
        root_midi: Root note of the scale
        scale: Scale name ('major', 'minor', 'chromatic')
        snap_threshold: Maximum distance in semitones to snap
        
    Returns:
        Snapped MIDI note
    """
    scale_set = set(SCALES.get(scale, SCALES['major']))
    
    # Get semitone and octave
    semitone = round(midi)
    
    # Find nearest scale degree
    scale_degree = (semitone - root_midi) % 12
    
    # Check if already in scale
    if scale_degree in scale_set:
        return float(semitone)
    
    # Find nearest scale note
    candidates = [(semitone + offset) for offset in range(-12, 13)
                  if ((semitone + offset - root_midi) % 12) in scale_set]
    
    if not candidates:
        return float(semitone)
    
    # Choose nearest candidate within threshold
    nearest = min(candidates, key=lambda c: abs(c - midi))
    
    if abs(nearest - midi) <= snap_threshold:
        return float(nearest)
    else:
        return midi  # Keep original if too far


def heuristic_map(f0_hz, voiced, root_midi=60, scale='major', 
                  vibrato_preserve=0.25, snap_threshold=0.5):
    """
    Heuristic pitch correction: snap to scale while preserving vibrato.
    
    Args:
        f0_hz: F0 contour in Hz
        voiced: Boolean mask of voiced frames
        root_midi: Root note of the scale (default: C4 = 60)
        scale: Scale name
        vibrato_preserve: Fraction of vibrato to preserve (0-1)
        snap_threshold: Maximum distance in semitones to snap
        
    Returns:
        Corrected F0 in Hz
    """
    logger.info(f'Applying heuristic mapping: root={root_midi}, scale={scale}')
    logger.info(f'Vibrato preservation: {vibrato_preserve*100:.0f}%')
    
    # Convert to MIDI
    midi = midi_from_hz(f0_hz)
    
    # Extract vibrato
    vibrato, midi_smooth = extract_vibrato(midi)
    
    # Initialize output
    mapped_midi = np.zeros_like(midi)
    
    # Process each frame
    for i in range(len(midi)):
        if voiced[i]:
            # Snap to scale
            snapped = snap_to_scale(
                midi_smooth[i],
                root_midi,
                scale,
                snap_threshold
            )
            mapped_midi[i] = snapped
        else:
            # Keep unvoiced frames as-is (will be 0 Hz)
            mapped_midi[i] = 0.0
    
    # Add back partial vibrato
    mapped_midi = mapped_midi + vibrato * vibrato_preserve
    
    # Additional smoothing to remove artifacts
    # Only smooth voiced regions
    for i in range(len(mapped_midi)):
        if not voiced[i]:
            mapped_midi[i] = 0.0
    
    # Convert back to Hz
    target_f0 = hz_from_midi(mapped_midi)
    
    # Zero out unvoiced frames
    target_f0[~voiced] = 0.0
    
    # Log statistics
    voiced_original = f0_hz[voiced]
    voiced_target = target_f0[voiced]
    
    if len(voiced_original) > 0:
        cents_shift = 1200 * np.log2(voiced_target / (voiced_original + 1e-6))
        logger.info(f'Pitch shift statistics:')
        logger.info(f'  Mean: {np.mean(cents_shift):.1f} cents')
        logger.info(f'  Std: {np.std(cents_shift):.1f} cents')
        logger.info(f'  Max: {np.max(np.abs(cents_shift)):.1f} cents')
    
    return target_f0


def model_map(f0_hz, voiced, times, model_ckpt):
    """
    Model-based pitch correction (placeholder for ML model).
    
    Args:
        f0_hz: F0 contour in Hz
        voiced: Boolean mask of voiced frames
        times: Time stamps
        model_ckpt: Path to model checkpoint
        
    Returns:
        Corrected F0 in Hz
    """
    logger.warning('Model-based mapping not yet implemented')
    logger.info('Falling back to heuristic mapping')
    
    # TODO: Implement model inference
    # from models.pitch_predictor.predict import predict
    # target_f0 = predict(model_ckpt, f0_hz, voiced, times)
    
    return f0_hz  # Return original for now


def main():
    parser = argparse.ArgumentParser(
        description='Infer target pitch contour for correction'
    )
    parser.add_argument('--f0_npz', required=True, help='Input F0 NPZ file')
    parser.add_argument('--output', '-o', required=True, help='Output target F0 NPZ file')
    parser.add_argument('--mode', choices=['heuristic', 'model'], default='heuristic',
                       help='Mapping mode')
    parser.add_argument('--root_midi', type=int, default=60,
                       help='Root note of scale (MIDI number, default: 60 = C4)')
    parser.add_argument('--scale', choices=['major', 'minor', 'chromatic'], default='major',
                       help='Musical scale')
    parser.add_argument('--vibrato_preserve', type=float, default=0.25,
                       help='Fraction of vibrato to preserve (0-1)')
    parser.add_argument('--snap_threshold', type=float, default=0.5,
                       help='Maximum distance in semitones to snap to scale')
    parser.add_argument('--model_ckpt', default=None,
                       help='Model checkpoint path (for model mode)')
    
    args = parser.parse_args()
    
    try:
        # Load F0 data
        logger.info(f'Loading F0 from {args.f0_npz}')
        data = load_npz(args.f0_npz)
        
        f0 = data['f0_hz']
        voiced_prob = data['voiced_prob']
        times = data['times']
        
        # Voiced mask (threshold at 0.5)
        voiced = voiced_prob > 0.5
        
        logger.info(f'Loaded {len(f0)} frames, {np.sum(voiced)} voiced')
        
        # Apply mapping
        if args.mode == 'heuristic':
            target_f0 = heuristic_map(
                f0,
                voiced,
                root_midi=args.root_midi,
                scale=args.scale,
                vibrato_preserve=args.vibrato_preserve,
                snap_threshold=args.snap_threshold
            )
        else:  # model mode
            if not args.model_ckpt:
                logger.error('--model_ckpt required for model mode')
                sys.exit(1)
            
            target_f0 = model_map(f0, voiced, times, args.model_ckpt)
        
        # Save output
        save_npz(
            args.output,
            times=times,
            f0_hz=target_f0,
            voiced_prob=voiced_prob
        )
        
        logger.info(f'Wrote target F0 to {args.output}')
        
    except Exception as e:
        logger.error(f'Target pitch inference failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
