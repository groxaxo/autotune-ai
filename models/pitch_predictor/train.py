"""Training script for pitch predictor model."""
import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from model import PitchPredictor
from dataset import PitchCorrectionDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PitchLoss(nn.Module):
    """Combined loss for pitch prediction."""
    
    def __init__(self, f0_weight=1.0, voicing_weight=0.5, perceptual_weight=0.1):
        super().__init__()
        self.f0_weight = f0_weight
        self.voicing_weight = voicing_weight
        self.perceptual_weight = perceptual_weight
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred, target, voiced_mask=None):
        """
        Compute loss.
        
        Args:
            pred: dict with 'f0_pred', 'voicing_prob', 'residual'
            target: dict with 'target_f0', 'voiced_prob'
            voiced_mask: optional mask for voiced frames
        """
        f0_pred = pred['f0_pred']
        voicing_pred = pred['voicing_prob']
        
        target_f0 = target['target_f0']
        target_voicing = target['voiced_prob']
        
        # F0 loss (only on voiced frames)
        if voiced_mask is None:
            voiced_mask = target_voicing > 0.5
        
        if voiced_mask.sum() > 0:
            f0_loss = self.mse_loss(f0_pred[voiced_mask], target_f0[voiced_mask])
        else:
            f0_loss = torch.tensor(0.0, device=f0_pred.device)
        
        # Voicing loss
        voicing_loss = self.bce_loss(voicing_pred, target_voicing)
        
        # Perceptual loss (in cents)
        # Cents = 1200 * log2(f_pred / f_target)
        if voiced_mask.sum() > 0:
            eps = 1e-6
            cents_diff = 1200 * torch.log2(
                (f0_pred[voiced_mask] + eps) / (target_f0[voiced_mask] + eps)
            )
            perceptual_loss = torch.mean(torch.abs(cents_diff))
        else:
            perceptual_loss = torch.tensor(0.0, device=f0_pred.device)
        
        # Total loss
        total_loss = (
            self.f0_weight * f0_loss +
            self.voicing_weight * voicing_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        return {
            'total': total_loss,
            'f0': f0_loss,
            'voicing': voicing_loss,
            'perceptual': perceptual_loss
        }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    f0_loss = 0
    voicing_loss = 0
    perceptual_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        mel_vocal = batch['mel_vocal'].to(device)
        mel_backing = batch['mel_backing'].to(device)
        f0_est = batch['f0_est'].to(device)
        target_f0 = batch['target_f0'].to(device)
        voiced_prob = batch['voiced_prob'].to(device)
        
        # Forward pass
        outputs = model(mel_backing, mel_vocal, f0_est)
        
        # Compute loss
        losses = criterion(
            outputs,
            {'target_f0': target_f0, 'voiced_prob': voiced_prob},
            voiced_mask=(voiced_prob > 0.5)
        )
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        f0_loss += losses['f0'].item()
        voicing_loss += losses['voicing'].item()
        perceptual_loss += losses['perceptual'].item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f'Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] '
                f'Loss: {losses["total"].item():.4f} '
                f'(F0: {losses["f0"].item():.4f}, '
                f'Voicing: {losses["voicing"].item():.4f}, '
                f'Perceptual: {losses["perceptual"].item():.2f} cents)'
            )
    
    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'f0': f0_loss / n_batches,
        'voicing': voicing_loss / n_batches,
        'perceptual': perceptual_loss / n_batches
    }


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    
    total_loss = 0
    f0_loss = 0
    voicing_loss = 0
    perceptual_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            mel_vocal = batch['mel_vocal'].to(device)
            mel_backing = batch['mel_backing'].to(device)
            f0_est = batch['f0_est'].to(device)
            target_f0 = batch['target_f0'].to(device)
            voiced_prob = batch['voiced_prob'].to(device)
            
            # Forward pass
            outputs = model(mel_backing, mel_vocal, f0_est)
            
            # Compute loss
            losses = criterion(
                outputs,
                {'target_f0': target_f0, 'voiced_prob': voiced_prob},
                voiced_mask=(voiced_prob > 0.5)
            )
            
            # Accumulate losses
            total_loss += losses['total'].item()
            f0_loss += losses['f0'].item()
            voicing_loss += losses['voicing'].item()
            perceptual_loss += losses['perceptual'].item()
    
    n_batches = len(dataloader)
    return {
        'total': total_loss / n_batches,
        'f0': f0_loss / n_batches,
        'voicing': voicing_loss / n_batches,
        'perceptual': perceptual_loss / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train pitch predictor model')
    
    # Data parameters
    parser.add_argument('--train_dir', required=True, help='Training data directory')
    parser.add_argument('--val_dir', default=None, help='Validation data directory')
    
    # Model parameters
    parser.add_argument('--mel_bins', type=int, default=80, help='Number of mel bins')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--segment_length', type=float, default=4.0, help='Segment length in seconds')
    
    # Loss weights
    parser.add_argument('--f0_weight', type=float, default=1.0, help='F0 loss weight')
    parser.add_argument('--voicing_weight', type=float, default=0.5, help='Voicing loss weight')
    parser.add_argument('--perceptual_weight', type=float, default=0.1, help='Perceptual loss weight')
    
    # Output
    parser.add_argument('--output_dir', default='checkpoints', help='Output directory')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Device
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    try:
        logger.info(f'Loading training data from {args.train_dir}')
        train_dataset = PitchCorrectionDataset(
            args.train_dir,
            segment_length=args.segment_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
        
        # Validation dataset
        if args.val_dir:
            logger.info(f'Loading validation data from {args.val_dir}')
            val_dataset = PitchCorrectionDataset(
                args.val_dir,
                segment_length=args.segment_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn
            )
        else:
            val_loader = None
            
    except Exception as e:
        logger.error(f'Failed to load datasets: {e}')
        sys.exit(1)
    
    # Create model
    logger.info('Creating model')
    model = PitchPredictor(
        mel_bins=args.mel_bins,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model parameters: {n_params:,}')
    
    # Create loss and optimizer
    criterion = PitchLoss(
        f0_weight=args.f0_weight,
        voicing_weight=args.voicing_weight,
        perceptual_weight=args.perceptual_weight
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f'\nEpoch {epoch}/{args.epochs}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(
            f'Train - Loss: {train_losses["total"]:.4f}, '
            f'F0: {train_losses["f0"]:.4f}, '
            f'Voicing: {train_losses["voicing"]:.4f}, '
            f'Perceptual: {train_losses["perceptual"]:.2f} cents'
        )
        
        # Validate
        if val_loader:
            val_losses = validate(model, val_loader, criterion, device)
            logger.info(
                f'Val - Loss: {val_losses["total"]:.4f}, '
                f'F0: {val_losses["f0"]:.4f}, '
                f'Voicing: {val_losses["voicing"]:.4f}, '
                f'Perceptual: {val_losses["perceptual"]:.2f} cents'
            )
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_path = output_dir / 'best.pt'
                torch.save(model.state_dict(), best_path)
                logger.info(f'Saved best model to {best_path}')
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['total'],
            }, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        # Update scheduler
        scheduler.step()
    
    # Save final model
    final_path = output_dir / 'final.pt'
    torch.save(model.state_dict(), final_path)
    logger.info(f'Saved final model to {final_path}')
    logger.info('Training complete!')


if __name__ == '__main__':
    main()
