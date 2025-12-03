"""Utility functions for audio I/O, path management, and data handling."""
import logging
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    logger.debug(f'Ensured directory exists: {directory}')


def read_wav(path, sr=None, mono=True):
    """
    Read audio file and return waveform and sample rate.
    
    Args:
        path: Path to audio file
        sr: Target sample rate (None = keep original)
        mono: Convert to mono if True
        
    Returns:
        y: Audio waveform as numpy array
        sr: Sample rate
    """
    try:
        y, file_sr = librosa.load(path, sr=sr, mono=mono)
        logger.debug(f'Loaded audio from {path}: shape={y.shape}, sr={file_sr}')
        return y, file_sr
    except Exception as e:
        logger.error(f'Failed to read audio from {path}: {e}')
        raise


def write_wav(path, y, sr, subtype='PCM_16'):
    """
    Write audio waveform to file.
    
    Args:
        path: Output path
        y: Audio waveform
        sr: Sample rate
        subtype: Audio format subtype
    """
    try:
        ensure_dir(os.path.dirname(path))
        # Normalize if clipping
        if np.abs(y).max() > 1.0:
            logger.warning(f'Audio clipping detected, normalizing before writing to {path}')
            y = y / np.abs(y).max() * 0.99
        sf.write(path, y, sr, subtype=subtype)
        logger.debug(f'Wrote audio to {path}: shape={y.shape}, sr={sr}')
    except Exception as e:
        logger.error(f'Failed to write audio to {path}: {e}')
        raise


def save_npz(path, **arrays):
    """
    Save numpy arrays to NPZ file.
    
    Args:
        path: Output path
        **arrays: Named arrays to save
    """
    try:
        ensure_dir(os.path.dirname(path))
        np.savez(path, **arrays)
        logger.debug(f'Saved NPZ to {path} with keys: {list(arrays.keys())}')
    except Exception as e:
        logger.error(f'Failed to save NPZ to {path}: {e}')
        raise


def load_npz(path):
    """
    Load numpy arrays from NPZ file.
    
    Args:
        path: Input path
        
    Returns:
        Dictionary-like object with arrays
    """
    try:
        data = np.load(path, allow_pickle=True)
        logger.debug(f'Loaded NPZ from {path} with keys: {list(data.keys())}')
        return data
    except Exception as e:
        logger.error(f'Failed to load NPZ from {path}: {e}')
        raise


def get_audio_duration(path):
    """Get duration of audio file in seconds."""
    try:
        duration = librosa.get_duration(path=path)
        logger.debug(f'Audio duration for {path}: {duration:.2f}s')
        return duration
    except Exception as e:
        logger.error(f'Failed to get duration for {path}: {e}')
        raise


def normalize_audio(y, target_db=-20.0):
    """
    Normalize audio to target dB level.
    
    Args:
        y: Audio waveform
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        y = y * gain
        logger.debug(f'Normalized audio: {current_db:.2f}dB -> {target_db:.2f}dB')
    return y


def safe_divide(a, b, fill_value=0.0):
    """Safe division with fill value for divide-by-zero."""
    result = np.full_like(a, fill_value, dtype=float)
    mask = b != 0
    result[mask] = a[mask] / b[mask]
    return result
