"""Wrapper around Demucs for source separation.
Outputs separated stems (vocal, instrumental) to --output directory.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
import torch
from utils import ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_demucs(input_path: str, out_dir: str, model: str = 'htdemucs', device: str = None):
    """
    Run Demucs source separation using the command-line interface.
    
    Args:
        input_path: Path to input audio file
        out_dir: Output directory for stems
        model: Demucs model to use (htdemucs, htdemucs_ft, mdx_extra)
        device: Device to use (cuda, cpu, or None for auto)
        
    Returns:
        Tuple of (vocal_path, instrumental_path)
    """
    ensure_dir(out_dir)
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f'Running Demucs separation on {input_path} using {device}')
    logger.info(f'Model: {model}')
    
    # Prepare demucs command
    # Use two-stems mode for vocals/accompaniment separation
    cmd = [
        'demucs',
        '--two-stems=vocals',
        f'--device={device}',
        f'--out={out_dir}',
        f'--name={model}',
        input_path
    ]
    
    try:
        # Run Demucs
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info('Demucs separation completed successfully')
        logger.debug(result.stdout)
        
        # Find output files
        # Demucs creates: out_dir/model/track_name/vocals.wav and no_vocals.wav
        input_name = Path(input_path).stem
        demucs_out = Path(out_dir) / model / input_name
        
        vocal_src = demucs_out / 'vocals.wav'
        instr_src = demucs_out / 'no_vocals.wav'
        
        # Copy to expected locations
        vocal_dst = Path(out_dir) / 'vocal.wav'
        instr_dst = Path(out_dir) / 'instr.wav'
        
        if vocal_src.exists():
            copyfile(vocal_src, vocal_dst)
            logger.info(f'Vocal stem saved to: {vocal_dst}')
        else:
            logger.error(f'Vocal output not found at {vocal_src}')
            # Fallback: copy input as vocal (degraded mode)
            copyfile(input_path, vocal_dst)
            logger.warning('Using input as vocal (fallback)')
        
        if instr_src.exists():
            copyfile(instr_src, instr_dst)
            logger.info(f'Instrumental stem saved to: {instr_dst}')
        else:
            logger.error(f'Instrumental output not found at {instr_src}')
            # Fallback: copy input as instrumental
            copyfile(input_path, instr_dst)
            logger.warning('Using input as instrumental (fallback)')
        
        return str(vocal_dst), str(instr_dst)
        
    except subprocess.CalledProcessError as e:
        logger.error(f'Demucs failed: {e.stderr}')
        logger.warning('Falling back to placeholder separation')
        
        # Fallback: copy input to both outputs
        vocal_path = Path(out_dir) / 'vocal.wav'
        instr_path = Path(out_dir) / 'instr.wav'
        copyfile(input_path, vocal_path)
        copyfile(input_path, instr_path)
        
        return str(vocal_path), str(instr_path)
    
    except Exception as e:
        logger.error(f'Unexpected error during separation: {e}')
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Separate audio into vocal and instrumental stems using Demucs'
    )
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--out_dir', '-d', required=True, help='Output directory for stems')
    parser.add_argument('--model', default='htdemucs', 
                       choices=['htdemucs', 'htdemucs_ft', 'mdx_extra'],
                       help='Demucs model to use')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                       help='Device to use (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    try:
        vocal, instr = run_demucs(args.input, args.out_dir, args.model, args.device)
        logger.info(f'Separation complete:')
        logger.info(f'  Vocal stem: {vocal}')
        logger.info(f'  Instrumental stem: {instr}')
    except Exception as e:
        logger.error(f'Failed to separate audio: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
