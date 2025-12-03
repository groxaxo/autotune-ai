"""Inference script for pitch predictor."""
import argparse
import logging
import sys
import torch
import numpy as np
import librosa

from model import PitchPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    logger.info(f'Loading model from {checkpoint_path}')
    
    model = PitchPredictor(
        mel_bins=80,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    )
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info('Model loaded successfully')
    return model


def compute_mel_spectrogram(audio, sr=44100, n_mels=80, hop_length=512):
    """Compute mel-spectrogram from audio."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=2048
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def predict_f0(model, vocal_audio, backing_audio, f0_est, sr=44100, hop_length=512, device='cpu'):
    """
    Predict target F0 using trained model.
    
    Args:
        model: Trained PitchPredictor model
        vocal_audio: Vocal audio waveform
        backing_audio: Backing track waveform (or zeros)
        f0_est: Estimated F0 contour
        sr: Sample rate
        hop_length: Hop length for mel-spectrogram
        device: Device to use
        
    Returns:
        Predicted F0 contour
    """
    logger.info('Computing mel-spectrograms...')
    
    # Compute mel-spectrograms
    mel_vocal = compute_mel_spectrogram(vocal_audio, sr, hop_length=hop_length)
    mel_backing = compute_mel_spectrogram(backing_audio, sr, hop_length=hop_length)
    
    # Match lengths
    min_frames = min(mel_vocal.shape[1], mel_backing.shape[1], len(f0_est))
    mel_vocal = mel_vocal[:, :min_frames]
    mel_backing = mel_backing[:, :min_frames]
    f0_est = f0_est[:min_frames]
    
    # Convert to tensors and add batch dimension
    mel_vocal = torch.from_numpy(mel_vocal).unsqueeze(0).float().to(device)
    mel_backing = torch.from_numpy(mel_backing).unsqueeze(0).float().to(device)
    f0_est_tensor = torch.from_numpy(f0_est).unsqueeze(0).float().to(device)
    
    logger.info(f'Input shapes: mel={mel_vocal.shape}, f0={f0_est_tensor.shape}')
    
    # Predict
    logger.info('Running model inference...')
    with torch.no_grad():
        outputs = model(mel_backing, mel_vocal, f0_est_tensor)
    
    # Extract F0 prediction
    f0_pred = outputs['f0_pred'].squeeze(0).cpu().numpy()
    voicing_prob = outputs['voicing_prob'].squeeze(0).cpu().numpy()
    
    # Apply voicing threshold
    f0_pred[voicing_prob < 0.5] = 0.0
    
    logger.info('Inference complete')
    return f0_pred, voicing_prob


def predict(ckpt_path, f0_est, voiced, times, vocal_audio=None, backing_audio=None, 
            sr=44100, hop_length=512, device=None):
    """
    High-level prediction function.
    
    Args:
        ckpt_path: Path to model checkpoint
        f0_est: Estimated F0 array
        voiced: Voiced probability array
        times: Time stamps
        vocal_audio: Optional vocal audio for mel computation
        backing_audio: Optional backing audio
        sr: Sample rate
        hop_length: Hop length
        device: Device to use
        
    Returns:
        Predicted target F0
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f'Using device: {device}')
    
    # Load model
    model = load_model(ckpt_path, device)
    
    # If audio not provided, return estimated F0 (no correction)
    if vocal_audio is None:
        logger.warning('No audio provided, returning estimated F0')
        return f0_est
    
    # If no backing, use zeros
    if backing_audio is None:
        backing_audio = np.zeros_like(vocal_audio)
    
    # Predict
    f0_pred, voicing_pred = predict_f0(
        model, vocal_audio, backing_audio, f0_est, sr, hop_length, device
    )
    
    # Post-process: smooth and blend with original where confidence is low
    from scipy.signal import medfilt
    f0_smooth = medfilt(f0_pred, kernel_size=5)
    
    # Blend with original based on voicing confidence
    blend_weight = np.clip(voicing_pred, 0.0, 1.0)
    f0_final = f0_smooth * blend_weight + f0_est[:len(f0_smooth)] * (1 - blend_weight)
    
    return f0_final


def main():
    parser = argparse.ArgumentParser(description='Predict target F0 using trained model')
    
    parser.add_argument('--ckpt', required=True, help='Model checkpoint path')
    parser.add_argument('--vocal', required=True, help='Vocal audio file')
    parser.add_argument('--backing', default=None, help='Backing track file (optional)')
    parser.add_argument('--f0_npz', required=True, help='Input F0 NPZ file')
    parser.add_argument('--output', required=True, help='Output NPZ file')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        # Setup device
        if args.device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        # Load audio
        logger.info(f'Loading vocal from {args.vocal}')
        vocal_audio, _ = librosa.load(args.vocal, sr=args.sr, mono=True)
        
        if args.backing:
            logger.info(f'Loading backing from {args.backing}')
            backing_audio, _ = librosa.load(args.backing, sr=args.sr, mono=True)
            # Match lengths
            min_len = min(len(vocal_audio), len(backing_audio))
            vocal_audio = vocal_audio[:min_len]
            backing_audio = backing_audio[:min_len]
        else:
            backing_audio = np.zeros_like(vocal_audio)
        
        # Load F0 data
        logger.info(f'Loading F0 from {args.f0_npz}')
        data = np.load(args.f0_npz)
        times = data['times']
        f0_est = data['f0_hz']
        voiced_prob = data['voiced_prob']
        
        # Predict
        f0_pred = predict(
            args.ckpt, f0_est, voiced_prob, times,
            vocal_audio, backing_audio, args.sr, device=device
        )
        
        # Save output
        np.savez(
            args.output,
            times=times[:len(f0_pred)],
            f0_hz=f0_pred,
            voiced_prob=voiced_prob[:len(f0_pred)]
        )
        
        logger.info(f'Predicted F0 saved to {args.output}')
        
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
