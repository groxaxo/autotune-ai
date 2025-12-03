"""F0 extraction script. Uses CREPE if available, otherwise falls back to librosa.pyin.
Saves arrays to an NPZ file with keys: times, f0_hz, voiced_prob
"""
import argparse
import logging
import sys
import numpy as np
import librosa
import torch
from utils import read_wav, save_npz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_f0_crepe(y, sr, hop_length=160, model_capacity='full', device=None):
    """
    Extract F0 using CREPE (deep learning-based pitch detection).
    
    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length in samples
        model_capacity: Model size (tiny, small, medium, large, full)
        device: Device to use (cuda/cpu, None for auto)
        
    Returns:
        Tuple of (times, f0_hz, voiced_prob)
    """
    try:
        import crepe
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use smaller model on CPU for speed
        if device == 'cpu' and model_capacity == 'full':
            model_capacity = 'small'
            logger.info('Using small model on CPU for faster inference')
        
        logger.info(f'Extracting F0 with CREPE (model={model_capacity}, device={device})')
        
        # CREPE expects step_size in milliseconds
        step_size = hop_length / sr * 1000  # Convert to ms
        
        # Run CREPE prediction
        time, frequency, confidence, activation = crepe.predict(
            y,
            sr,
            viterbi=True,
            step_size=step_size,
            model_capacity=model_capacity,
            device=device
        )
        
        # Set unvoiced regions to 0 Hz (low confidence)
        f0 = frequency.copy()
        f0[confidence < 0.5] = 0.0
        
        # Apply median filter for smoothing
        from scipy.signal import medfilt
        f0_smoothed = medfilt(f0, kernel_size=3)
        
        # Preserve original f0 where voiced
        f0_final = np.where(confidence > 0.5, f0_smoothed, 0.0)
        
        logger.info(f'CREPE extraction complete: {len(time)} frames')
        return time, f0_final, confidence
        
    except ImportError:
        logger.error('CREPE not installed')
        raise
    except Exception as e:
        logger.error(f'CREPE extraction failed: {e}')
        raise


def extract_f0_librosa(y, sr, hop_length=512, fmin=None, fmax=None):
    """
    Extract F0 using librosa's pyin algorithm.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length in samples
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        
    Returns:
        Tuple of (times, f0_hz, voiced_prob)
    """
    logger.info('Extracting F0 with librosa pyin')
    
    # Default frequency range for vocals
    if fmin is None:
        fmin = librosa.note_to_hz('C2')  # ~65 Hz
    if fmax is None:
        fmax = librosa.note_to_hz('C7')  # ~2093 Hz
    
    # Extract pitch using pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length
    )
    
    # Compute time stamps
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )
    
    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_probs = np.nan_to_num(voiced_probs, nan=0.0)
    
    logger.info(f'Librosa pyin extraction complete: {len(times)} frames')
    return times, f0, voiced_probs


def main():
    parser = argparse.ArgumentParser(
        description='Extract pitch (F0) from audio using CREPE or librosa'
    )
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', required=True, help='Output NPZ file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--method', choices=['crepe', 'librosa'], default='crepe',
                       help='Pitch extraction method')
    parser.add_argument('--hop_length', type=int, default=None,
                       help='Hop length in samples (default: method-specific)')
    parser.add_argument('--model_capacity', default='full',
                       choices=['tiny', 'small', 'medium', 'large', 'full'],
                       help='CREPE model capacity')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--fmin', type=float, default=None,
                       help='Minimum frequency in Hz')
    parser.add_argument('--fmax', type=float, default=None,
                       help='Maximum frequency in Hz')
    
    args = parser.parse_args()
    
    try:
        # Load audio
        logger.info(f'Loading audio from {args.input}')
        y, sr = read_wav(args.input, sr=args.sr)
        
        # Extract F0
        if args.method == 'crepe':
            try:
                hop_length = args.hop_length or 160
                times, f0, voiced = extract_f0_crepe(
                    y, sr,
                    hop_length=hop_length,
                    model_capacity=args.model_capacity,
                    device=args.device
                )
            except Exception as e:
                logger.warning(f'CREPE failed: {e}, falling back to librosa')
                hop_length = args.hop_length or 512
                times, f0, voiced = extract_f0_librosa(
                    y, sr,
                    hop_length=hop_length,
                    fmin=args.fmin,
                    fmax=args.fmax
                )
        else:
            hop_length = args.hop_length or 512
            times, f0, voiced = extract_f0_librosa(
                y, sr,
                hop_length=hop_length,
                fmin=args.fmin,
                fmax=args.fmax
            )
        
        # Save to NPZ
        save_npz(
            args.output,
            times=times,
            f0_hz=f0,
            voiced_prob=voiced
        )
        
        # Log statistics
        voiced_mask = voiced > 0.5
        if np.any(voiced_mask):
            f0_voiced = f0[voiced_mask]
            logger.info(f'F0 statistics:')
            logger.info(f'  Voiced frames: {np.sum(voiced_mask)} / {len(voiced)} ({100*np.mean(voiced_mask):.1f}%)')
            logger.info(f'  F0 range: {f0_voiced.min():.1f} - {f0_voiced.max():.1f} Hz')
            logger.info(f'  F0 mean: {f0_voiced.mean():.1f} Hz')
        
        logger.info(f'Saved F0 to {args.output}')
        
    except Exception as e:
        logger.error(f'F0 extraction failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
