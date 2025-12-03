"""PyTorch model for pitch prediction.
Architecture: CNN + Transformer for processing mel-spectrograms and F0 estimates.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class PitchPredictor(nn.Module):
    """
    Pitch prediction model using CNN + Transformer architecture.
    
    Inputs:
        - mel_backing: Mel-spectrogram of backing track [B, mel_bins, T]
        - mel_vocal: Mel-spectrogram of vocal [B, mel_bins, T]
        - f0_est: Estimated F0 contour [B, T]
    
    Outputs:
        - f0_pred: Predicted target F0 [B, T]
        - voicing_prob: Voicing probability [B, T]
    """
    
    def __init__(self, mel_bins=80, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.mel_bins = mel_bins
        self.hidden_dim = hidden_dim
        
        # CNN front-end for backing track
        self.cnn_backing = nn.Sequential(
            nn.Conv1d(mel_bins, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CNN front-end for vocal
        self.cnn_vocal = nn.Sequential(
            nn.Conv1d(mel_bins, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # F0 embedding
        self.f0_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.f0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.voicing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Residual correction head (optional)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, mel_backing, mel_vocal, f0_est, mask=None):
        """
        Forward pass.
        
        Args:
            mel_backing: [B, mel_bins, T] backing mel-spectrogram
            mel_vocal: [B, mel_bins, T] vocal mel-spectrogram
            f0_est: [B, T] estimated F0
            mask: [B, T] optional padding mask
            
        Returns:
            dict with keys:
                - f0_pred: [B, T] predicted F0
                - voicing_prob: [B, T] voicing probability
                - residual: [B, T] pitch residual
        """
        B, _, T = mel_backing.shape
        
        # Extract features from backing and vocal
        backing_feat = self.cnn_backing(mel_backing)  # [B, hidden_dim, T]
        vocal_feat = self.cnn_vocal(mel_vocal)  # [B, hidden_dim, T]
        
        # Embed F0
        f0_feat = self.f0_embed(f0_est.unsqueeze(-1))  # [B, T, hidden_dim]
        
        # Transpose CNN features for concatenation
        backing_feat = backing_feat.transpose(1, 2)  # [B, T, hidden_dim]
        vocal_feat = vocal_feat.transpose(1, 2)  # [B, T, hidden_dim]
        
        # Fuse features
        fused = torch.cat([backing_feat, vocal_feat, f0_feat], dim=-1)  # [B, T, hidden_dim*3]
        fused = self.fusion(fused)  # [B, T, hidden_dim]
        
        # Add positional encoding
        fused = self.pos_encoder(fused)
        
        # Transformer encoding
        encoded = self.transformer(fused, src_key_padding_mask=mask)  # [B, T, hidden_dim]
        
        # Predict outputs
        f0_pred = self.f0_head(encoded).squeeze(-1)  # [B, T]
        voicing_prob = self.voicing_head(encoded).squeeze(-1)  # [B, T]
        residual = self.residual_head(encoded).squeeze(-1)  # [B, T]
        
        return {
            'f0_pred': f0_pred,
            'voicing_prob': voicing_prob,
            'residual': residual
        }


if __name__ == '__main__':
    # Test model
    model = PitchPredictor(mel_bins=80, hidden_dim=256)
    
    # Create dummy inputs
    B, T, mel_bins = 2, 100, 80
    mel_backing = torch.randn(B, mel_bins, T)
    mel_vocal = torch.randn(B, mel_bins, T)
    f0_est = torch.randn(B, T) * 100 + 200  # Random F0 around 200 Hz
    
    # Forward pass
    outputs = model(mel_backing, mel_vocal, f0_est)
    
    print('Model test:')
    print(f'  Input shape: mel={mel_backing.shape}, f0={f0_est.shape}')
    print(f'  Output shapes:')
    print(f'    f0_pred: {outputs["f0_pred"].shape}')
    print(f'    voicing_prob: {outputs["voicing_prob"].shape}')
    print(f'    residual: {outputs["residual"].shape}')
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {n_params:,}')
