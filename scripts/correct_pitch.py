"""Apply pitch correction using WORLD vocoder or PSOLA fallback.
Inputs: original vocal wav, target npz (times, f0_hz)
"""
import argparse
import logging
import sys
import numpy as np
import librosa
from utils import read_wav, write_wav, load_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def correct_with_world(y, sr, times_target, f0_target):
    """
    Apply pitch correction using WORLD vocoder.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        times_target: Target time stamps
        f0_target: Target F0 contour in Hz
        
    Returns:
        Corrected audio waveform
    """
    try:
        import pyworld as pw
        
        logger.info('Applying pitch correction with WORLD vocoder')
        
        # Ensure float64 for WORLD
        y_float = y.astype(np.float64)
        
        # Extract F0 using DIO + StoneMask
        logger.info('Extracting F0 with DIO...')
        _f0, t = pw.dio(y_float, sr)
        _f0 = pw.stonemask(y_float, _f0, t, sr)
        
        # Extract spectral envelope
        logger.info('Extracting spectral envelope...')
        sp = pw.cheaptrick(y_float, _f0, t, sr)
        
        # Extract aperiodicity
        logger.info('Extracting aperiodicity...')
        ap = pw.d4c(y_float, _f0, t, sr)
        
        # Interpolate target F0 to match WORLD time grid
        logger.info('Interpolating target F0 to WORLD time grid...')
        f0_new = np.interp(t, times_target, f0_target, left=0.0, right=0.0)
        
        # Ensure voiced frames stay voiced
        # If original was voiced but target is 0, keep a minimal value
        voiced_original = _f0 > 0
        voiced_target = f0_new > 0
        
        # Preserve voicing decisions from target
        f0_new[~voiced_target] = 0.0
        
        logger.info(f'F0 statistics:')
        logger.info(f'  Original voiced frames: {np.sum(voiced_original)}')
        logger.info(f'  Target voiced frames: {np.sum(voiced_target)}')
        
        # Synthesize with corrected F0
        logger.info('Synthesizing corrected audio...')
        y_synth = pw.synthesize(f0_new, sp, ap, sr)
        
        # Convert back to float32 and normalize
        y_synth = y_synth.astype(np.float32)
        
        # Normalize to prevent clipping
        max_val = np.abs(y_synth).max()
        if max_val > 0:
            y_synth = y_synth / max_val * 0.99
        
        logger.info('WORLD vocoder synthesis complete')
        return y_synth
        
    except ImportError:
        logger.error('pyworld not installed')
        raise
    except Exception as e:
        logger.error(f'WORLD vocoder failed: {e}')
        raise


def correct_with_psola(y, sr, times_target, f0_target):
    """
    Apply pitch correction using PSOLA (Pitch Synchronous Overlap-Add).
    This is a fallback method using librosa.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        times_target: Target time stamps
        f0_target: Target F0 contour in Hz
        
    Returns:
        Corrected audio waveform (or original if fails)
    """
    logger.warning('PSOLA fallback is limited - returning original with simple pitch shift')
    
    try:
        # Extract original F0 for comparison
        f0_original, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_original = np.nan_to_num(f0_original)
        
        # Compute average pitch shift
        voiced_mask = (f0_original > 0) & (f0_target > 0)
        
        if np.sum(voiced_mask) > 0:
            # Interpolate target to match original F0 times
            times_original = librosa.frames_to_time(
                np.arange(len(f0_original)),
                sr=sr,
                hop_length=512
            )
            f0_target_interp = np.interp(
                times_original,
                times_target,
                f0_target,
                left=0.0,
                right=0.0
            )
            
            # Compute average shift in semitones
            ratio = f0_target_interp[voiced_mask] / f0_original[voiced_mask]
            n_steps = np.median(12 * np.log2(ratio))
            
            logger.info(f'Average pitch shift: {n_steps:.2f} semitones')
            
            # Apply pitch shift
            y_shifted = librosa.effects.pitch_shift(
                y,
                sr=sr,
                n_steps=n_steps
            )
            
            return y_shifted
        else:
            logger.warning('No voiced frames found, returning original')
            return y
            
    except Exception as e:
        logger.error(f'PSOLA fallback failed: {e}')
        return y


def main():
    parser = argparse.ArgumentParser(
        description='Apply pitch correction to vocal audio'
    )
    parser.add_argument('--vocal', '-v', required=True, help='Input vocal file')
    parser.add_argument('--target_npz', required=True, help='Target F0 NPZ file')
    parser.add_argument('--output', '-o', required=True, help='Output corrected file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--method', choices=['world', 'psola'], default='world',
                       help='Pitch correction method')
    
    args = parser.parse_args()
    
    try:
        # Load vocal
        logger.info(f'Loading vocal from {args.vocal}')
        y, sr = read_wav(args.vocal, sr=args.sr)
        
        # Load target F0
        logger.info(f'Loading target F0 from {args.target_npz}')
        data = load_npz(args.target_npz)
        times_target = data['times']
        f0_target = data['f0_hz']
        
        logger.info(f'Target F0 range: {f0_target[f0_target > 0].min():.1f} - {f0_target[f0_target > 0].max():.1f} Hz')
        
        # Apply correction
        if args.method == 'world':
            try:
                y_out = correct_with_world(y, sr, times_target, f0_target)
            except Exception as e:
                logger.warning(f'WORLD failed: {e}')
                logger.info('Falling back to PSOLA')
                y_out = correct_with_psola(y, sr, times_target, f0_target)
        else:
            y_out = correct_with_psola(y, sr, times_target, f0_target)
        
        # Save output
        write_wav(args.output, y_out, sr)
        logger.info(f'Wrote corrected vocal to {args.output}')
        
    except Exception as e:
        logger.error(f'Pitch correction failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
