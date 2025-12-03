"""Preprocess audio: denoising, VAD, and optional time alignment.
Outputs cleaned vocal wav.
"""
import argparse
import logging
import sys
import numpy as np
import librosa
from utils import read_wav, write_wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def denoise(y, sr, stationary=True, prop_decrease=0.8):
    """
    Apply noise reduction to audio using noisereduce.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        stationary: Assume stationary noise
        prop_decrease: Proportion of noise to reduce (0-1)
        
    Returns:
        Denoised audio
    """
    try:
        import noisereduce as nr
        
        # Normalize input to prevent clipping
        y_norm = y / (np.abs(y).max() + 1e-8)
        
        logger.info(f'Applying noise reduction (prop_decrease={prop_decrease})')
        reduced = nr.reduce_noise(
            y=y_norm,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        
        # Restore original scale
        reduced = reduced * (np.abs(y).max() + 1e-8)
        
        logger.info('Noise reduction complete')
        return reduced
        
    except ImportError:
        logger.warning('noisereduce not installed, skipping denoising')
        return y
    except Exception as e:
        logger.error(f'Denoising failed: {e}')
        logger.warning('Returning original audio')
        return y


def apply_vad(y, sr, top_db=30, frame_length=2048, hop_length=512):
    """
    Apply voice activity detection to remove silence.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        top_db: Threshold in dB below reference to consider as silence
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Audio with silence removed
    """
    try:
        logger.info(f'Applying VAD (top_db={top_db})')
        
        # Split audio at silence
        intervals = librosa.effects.split(
            y,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        if len(intervals) == 0:
            logger.warning('No voice activity detected, returning original')
            return y
        
        # Concatenate non-silent intervals
        segments = [y[start:end] for start, end in intervals]
        y_voiced = np.concatenate(segments)
        
        logger.info(f'VAD complete: {len(y)} -> {len(y_voiced)} samples')
        return y_voiced
        
    except Exception as e:
        logger.error(f'VAD failed: {e}')
        logger.warning('Returning original audio')
        return y


def align_to_backing(vocal_y, backing_y, sr):
    """
    Align vocal to backing track using cross-correlation.
    
    Args:
        vocal_y: Vocal waveform
        backing_y: Backing track waveform
        sr: Sample rate
        
    Returns:
        Aligned vocal waveform
    """
    try:
        logger.info('Computing time alignment using onset correlation')
        
        # Compute onset strength envelopes
        vocal_onset = librosa.onset.onset_strength(y=vocal_y, sr=sr)
        backing_onset = librosa.onset.onset_strength(y=backing_y, sr=sr)
        
        # Find offset using cross-correlation
        correlation = np.correlate(backing_onset, vocal_onset, mode='full')
        offset_frames = np.argmax(correlation) - len(vocal_onset) + 1
        
        # Convert to samples
        hop_length = 512  # Default hop length
        offset_samples = offset_frames * hop_length
        
        logger.info(f'Detected offset: {offset_samples} samples ({offset_samples/sr:.3f}s)')
        
        # Align vocal
        if offset_samples > 0:
            # Vocal starts after backing - pad start
            aligned = np.pad(vocal_y, (offset_samples, 0), mode='constant')
        elif offset_samples < 0:
            # Vocal starts before backing - trim start
            aligned = vocal_y[-offset_samples:]
        else:
            aligned = vocal_y
        
        # Match length to backing
        if len(aligned) > len(backing_y):
            aligned = aligned[:len(backing_y)]
        elif len(aligned) < len(backing_y):
            aligned = np.pad(aligned, (0, len(backing_y) - len(aligned)), mode='constant')
        
        logger.info(f'Alignment complete: final length {len(aligned)} samples')
        return aligned
        
    except Exception as e:
        logger.error(f'Alignment failed: {e}')
        logger.warning('Returning original vocal')
        return vocal_y


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess audio with denoising, VAD, and alignment'
    )
    parser.add_argument('--vocal', '-v', required=True, help='Input vocal file')
    parser.add_argument('--backing', '-b', default=None, help='Backing track for alignment')
    parser.add_argument('--output', '-o', required=True, help='Output preprocessed file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--denoise', action='store_true', help='Apply noise reduction')
    parser.add_argument('--vad', action='store_true', help='Apply voice activity detection')
    parser.add_argument('--align', action='store_true', help='Align to backing track')
    parser.add_argument('--prop_decrease', type=float, default=0.8,
                       help='Noise reduction strength (0-1)')
    parser.add_argument('--top_db', type=float, default=30,
                       help='VAD silence threshold in dB')
    
    args = parser.parse_args()
    
    try:
        # Load vocal
        logger.info(f'Loading vocal from {args.vocal}')
        y, sr = read_wav(args.vocal, sr=args.sr)
        
        # Apply denoising
        if args.denoise:
            y = denoise(y, sr, prop_decrease=args.prop_decrease)
        
        # Apply VAD
        if args.vad:
            y = apply_vad(y, sr, top_db=args.top_db)
        
        # Apply alignment
        if args.align:
            if not args.backing:
                logger.error('--backing required for alignment')
                sys.exit(1)
            
            logger.info(f'Loading backing track from {args.backing}')
            backing_y, _ = read_wav(args.backing, sr=sr)
            y = align_to_backing(y, backing_y, sr)
        
        # Save output
        write_wav(args.output, y, sr)
        logger.info(f'Preprocessed audio saved to {args.output}')
        
    except Exception as e:
        logger.error(f'Preprocessing failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
