"""Dataset for pitch predictor model training."""
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
from pathlib import Path


class PitchCorrectionDataset(Dataset):
    """
    Dataset for pitch correction model.
    
    Expected data structure:
        data_dir/
            pair_0001/
                detuned_vocal.wav
                target.npz  (with keys: times, f0_hz, voiced_prob)
                backing.wav (optional)
            pair_0002/
                ...
    """
    
    def __init__(self, data_dir, sr=44100, n_mels=80, hop_length=512, 
                 segment_length=4.0, transform=None):
        """
        Args:
            data_dir: Root directory containing paired data
            sr: Sample rate
            n_mels: Number of mel bins
            hop_length: Hop length for STFT
            segment_length: Length of segments in seconds
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.segment_length = segment_length
        self.segment_frames = int(segment_length * sr / hop_length)
        self.transform = transform
        
        # Find all pairs
        self.pairs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(self.pairs) == 0:
            raise ValueError(f'No data found in {data_dir}')
        
        print(f'Found {len(self.pairs)} data pairs in {data_dir}')
    
    def __len__(self):
        return len(self.pairs)
    
    def _compute_mel(self, audio):
        """Compute mel-spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=2048
        )
        # Convert to log scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    
    def _load_audio(self, path):
        """Load audio file."""
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio
    
    def _load_f0(self, npz_path):
        """Load F0 data from NPZ."""
        data = np.load(npz_path)
        return data['times'], data['f0_hz'], data['voiced_prob']
    
    def _segment_data(self, mel, f0, voiced_prob):
        """Extract random segment from data."""
        T = mel.shape[1]
        
        if T <= self.segment_frames:
            # Pad if too short
            pad_frames = self.segment_frames - T
            mel = np.pad(mel, ((0, 0), (0, pad_frames)), mode='constant')
            f0 = np.pad(f0, (0, pad_frames), mode='constant')
            voiced_prob = np.pad(voiced_prob, (0, pad_frames), mode='constant')
            start_idx = 0
        else:
            # Random crop
            start_idx = np.random.randint(0, T - self.segment_frames)
        
        end_idx = start_idx + self.segment_frames
        
        mel_seg = mel[:, start_idx:end_idx]
        f0_seg = f0[start_idx:end_idx]
        voiced_seg = voiced_prob[start_idx:end_idx]
        
        return mel_seg, f0_seg, voiced_seg
    
    def __getitem__(self, idx):
        """Get a training sample."""
        pair_dir = self.pairs[idx]
        
        # Load detuned vocal
        detuned_path = pair_dir / 'detuned_vocal.wav'
        detuned_audio = self._load_audio(detuned_path)
        
        # Load target F0
        target_path = pair_dir / 'target.npz'
        times, target_f0, voiced_prob = self._load_f0(target_path)
        
        # Load backing track if available
        backing_path = pair_dir / 'backing.wav'
        if backing_path.exists():
            backing_audio = self._load_audio(backing_path)
            # Match length
            min_len = min(len(detuned_audio), len(backing_audio))
            detuned_audio = detuned_audio[:min_len]
            backing_audio = backing_audio[:min_len]
        else:
            # Use zeros if no backing track
            backing_audio = np.zeros_like(detuned_audio)
        
        # Compute mel-spectrograms
        mel_vocal = self._compute_mel(detuned_audio)
        mel_backing = self._compute_mel(backing_audio)
        
        # Extract estimated F0 (could be noisy/detuned)
        # For training, we use librosa to extract F0 from detuned audio
        f0_est, _, _ = librosa.pyin(
            detuned_audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
            hop_length=self.hop_length
        )
        f0_est = np.nan_to_num(f0_est, nan=0.0)
        
        # Resample target F0 to match mel time grid
        mel_times = librosa.frames_to_time(
            np.arange(mel_vocal.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length
        )
        target_f0_resampled = np.interp(mel_times, times, target_f0)
        voiced_prob_resampled = np.interp(mel_times, times, voiced_prob)
        
        # Match F0 length to mel
        min_frames = min(len(f0_est), len(target_f0_resampled), mel_vocal.shape[1])
        f0_est = f0_est[:min_frames]
        target_f0_resampled = target_f0_resampled[:min_frames]
        voiced_prob_resampled = voiced_prob_resampled[:min_frames]
        mel_vocal = mel_vocal[:, :min_frames]
        mel_backing = mel_backing[:, :min_frames]
        
        # Segment data
        mel_vocal, f0_est, voiced_seg = self._segment_data(
            mel_vocal, f0_est, voiced_prob_resampled
        )
        mel_backing, target_f0_seg, _ = self._segment_data(
            mel_backing, target_f0_resampled, voiced_prob_resampled
        )
        
        # Convert to tensors
        sample = {
            'mel_vocal': torch.from_numpy(mel_vocal).float(),
            'mel_backing': torch.from_numpy(mel_backing).float(),
            'f0_est': torch.from_numpy(f0_est).float(),
            'target_f0': torch.from_numpy(target_f0_seg).float(),
            'voiced_prob': torch.from_numpy(voiced_seg).float()
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'mel_vocal': torch.stack([item['mel_vocal'] for item in batch]),
        'mel_backing': torch.stack([item['mel_backing'] for item in batch]),
        'f0_est': torch.stack([item['f0_est'] for item in batch]),
        'target_f0': torch.stack([item['target_f0'] for item in batch]),
        'voiced_prob': torch.stack([item['voiced_prob'] for item in batch])
    }


if __name__ == '__main__':
    # Test dataset
    print('Dataset test - requires data in data/train/')
    try:
        dataset = PitchCorrectionDataset('data/train', segment_length=2.0)
        print(f'Dataset size: {len(dataset)}')
        
        if len(dataset) > 0:
            sample = dataset[0]
            print('Sample shapes:')
            for key, val in sample.items():
                print(f'  {key}: {val.shape}')
    except Exception as e:
        print(f'Dataset test failed: {e}')
        print('This is expected if training data is not available')
