"""Post-processing: loudness normalization, de-essing, simple EQ, and mixing back with backing track.
"""
import argparse
import logging
import sys
import numpy as np
from scipy.signal import butter, sosfilt
from utils import read_wav, write_wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_lufs(y, sr, target_lufs=-14.0):
    """
    Normalize audio to target LUFS loudness level.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized audio
    """
    try:
        import pyloudnorm as pyln
        
        logger.info(f'Normalizing to {target_lufs} LUFS')
        
        # Create loudness meter
        meter = pyln.Meter(sr)
        
        # Measure loudness
        loudness = meter.integrated_loudness(y)
        logger.info(f'Current loudness: {loudness:.1f} LUFS')
        
        # Normalize
        y_normalized = pyln.normalize.loudness(y, loudness, target_lufs)
        
        # Prevent clipping
        max_val = np.abs(y_normalized).max()
        if max_val > 1.0:
            y_normalized = y_normalized / max_val * 0.99
            logger.warning('Clipping prevented after normalization')
        
        logger.info(f'Normalized to {target_lufs} LUFS')
        return y_normalized
        
    except ImportError:
        logger.warning('pyloudnorm not installed, using RMS normalization')
        return normalize_rms(y, target_db=-20.0)
    except Exception as e:
        logger.error(f'LUFS normalization failed: {e}')
        logger.warning('Falling back to RMS normalization')
        return normalize_rms(y, target_db=-20.0)


def normalize_rms(y, target_db=-20.0):
    """
    Normalize audio to target RMS level (fallback method).
    
    Args:
        y: Audio waveform
        target_db: Target RMS level in dB
        
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(y**2))
    
    if rms > 0:
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        y_normalized = y * gain
        
        # Prevent clipping
        max_val = np.abs(y_normalized).max()
        if max_val > 1.0:
            y_normalized = y_normalized / max_val * 0.99
        
        logger.info(f'RMS normalized: {current_db:.1f}dB -> {target_db:.1f}dB')
        return y_normalized
    
    return y


def de_ess(y, sr, freq=5000, q=0.7, gain_db=-6.0):
    """
    Apply simple de-essing by attenuating high frequencies.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        freq: Frequency to attenuate (Hz)
        q: Q factor (bandwidth)
        gain_db: Gain reduction in dB
        
    Returns:
        De-essed audio
    """
    try:
        logger.info(f'Applying de-esser at {freq} Hz')
        
        # Design high-shelf filter for attenuation
        nyquist = sr / 2
        norm_freq = freq / nyquist
        
        # Create a simple high-pass filter to detect sibilance
        sos_detect = butter(2, norm_freq, btype='high', output='sos')
        high_freq = sosfilt(sos_detect, y)
        
        # Compute envelope of high-frequency content
        envelope = np.abs(high_freq)
        
        # Smooth envelope
        from scipy.ndimage import uniform_filter1d
        envelope_smooth = uniform_filter1d(envelope, size=int(sr * 0.01))  # 10ms window
        
        # Compute threshold (e.g., 70th percentile)
        threshold = np.percentile(envelope_smooth, 70)
        
        # Create gain reduction mask
        gain_reduction = np.ones_like(y)
        sibilant_mask = envelope_smooth > threshold
        
        if np.any(sibilant_mask):
            gain_linear = 10 ** (gain_db / 20)
            gain_reduction[sibilant_mask] = gain_linear
            
            # Smooth gain changes
            gain_reduction = uniform_filter1d(gain_reduction, size=int(sr * 0.005))  # 5ms
            
            # Apply gain reduction
            y_deessed = y * gain_reduction
            
            logger.info(f'De-essing applied to {np.sum(sibilant_mask)} samples')
            return y_deessed
        else:
            logger.info('No sibilance detected, skipping de-essing')
            return y
            
    except Exception as e:
        logger.error(f'De-essing failed: {e}')
        return y


def mix_vocal_backing(vocal_y, backing_y, vocal_gain_db=0.0, backing_gain_db=0.0):
    """
    Mix vocal and backing tracks.
    
    Args:
        vocal_y: Vocal waveform
        backing_y: Backing track waveform
        vocal_gain_db: Vocal gain in dB
        backing_gain_db: Backing gain in dB
        
    Returns:
        Mixed audio
    """
    logger.info('Mixing vocal and backing tracks')
    logger.info(f'Vocal gain: {vocal_gain_db:+.1f} dB')
    logger.info(f'Backing gain: {backing_gain_db:+.1f} dB')
    
    # Match lengths
    min_len = min(len(vocal_y), len(backing_y))
    vocal_y = vocal_y[:min_len]
    backing_y = backing_y[:min_len]
    
    # Apply gains
    vocal_gain = 10 ** (vocal_gain_db / 20.0)
    backing_gain = 10 ** (backing_gain_db / 20.0)
    
    # Mix
    mixed = vocal_y * vocal_gain + backing_y * backing_gain
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 1.0:
        mixed = mixed / max_val * 0.99
        logger.warning('Mixed audio normalized to prevent clipping')
    
    logger.info(f'Mixed audio length: {len(mixed)} samples ({len(mixed)/44100:.2f}s)')
    
    return mixed


def main():
    parser = argparse.ArgumentParser(
        description='Post-process and mix corrected vocal with backing track'
    )
    parser.add_argument('--vocal', '-v', required=True, help='Corrected vocal file')
    parser.add_argument('--backing', '-b', required=True, help='Backing track file')
    parser.add_argument('--output', '-o', required=True, help='Output mixed file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--vocal_gain_db', type=float, default=0.0,
                       help='Vocal gain in dB')
    parser.add_argument('--backing_gain_db', type=float, default=0.0,
                       help='Backing gain in dB')
    parser.add_argument('--target_lufs', type=float, default=-14.0,
                       help='Target loudness in LUFS')
    parser.add_argument('--deess', action='store_true',
                       help='Apply de-essing to vocal')
    parser.add_argument('--deess_freq', type=float, default=5000,
                       help='De-esser frequency in Hz')
    
    args = parser.parse_args()
    
    try:
        # Load vocal
        logger.info(f'Loading vocal from {args.vocal}')
        vocal_y, sr = read_wav(args.vocal, sr=args.sr)
        
        # Load backing
        logger.info(f'Loading backing track from {args.backing}')
        backing_y, _ = read_wav(args.backing, sr=sr)
        
        # Normalize vocal
        vocal_y = normalize_lufs(vocal_y, sr, target_lufs=args.target_lufs)
        
        # Apply de-essing if requested
        if args.deess:
            vocal_y = de_ess(vocal_y, sr, freq=args.deess_freq)
        
        # Mix
        mixed = mix_vocal_backing(
            vocal_y,
            backing_y,
            vocal_gain_db=args.vocal_gain_db,
            backing_gain_db=args.backing_gain_db
        )
        
        # Save output
        write_wav(args.output, mixed, sr)
        logger.info(f'Postprocessed and mixed audio saved to {args.output}')
        
    except Exception as e:
        logger.error(f'Post-processing failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
