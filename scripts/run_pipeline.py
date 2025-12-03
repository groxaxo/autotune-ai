"""Main pipeline orchestration script for autotune processing.
Coordinates all steps: separation, preprocessing, pitch extraction, correction, and mixing.
"""
import argparse
import logging
import sys
import os
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a subprocess command and handle errors."""
    logger.info(f'Running: {description}')
    logger.debug(f'Command: {" ".join(cmd)}')
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f'{description} failed:')
        logger.error(e.stderr)
        return False


def run_fast_pipeline(input_path, output_path, work_dir, args):
    """
    Run the fast pipeline using heuristic pitch correction.
    
    Pipeline steps:
    1. Separate vocals from backing (if not already separated)
    2. Preprocess vocal (optional: denoise, VAD, align)
    3. Extract F0
    4. Infer target F0 (heuristic)
    5. Correct pitch
    6. Post-process and mix
    """
    logger.info('=== Running FAST pipeline ===')
    
    # Create work directories
    sep_dir = Path(work_dir) / 'separation'
    pre_dir = Path(work_dir) / 'preprocessed'
    f0_dir = Path(work_dir) / 'f0'
    target_dir = Path(work_dir) / 'target'
    corrected_dir = Path(work_dir) / 'corrected'
    
    for d in [sep_dir, pre_dir, f0_dir, target_dir, corrected_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Separation
    if args.vocal and args.backing:
        # Pre-separated inputs provided
        logger.info('Using pre-separated vocal and backing tracks')
        vocal_path = args.vocal
        backing_path = args.backing
    else:
        # Need to separate
        logger.info('Step 1/6: Source separation')
        vocal_path = sep_dir / 'vocal.wav'
        backing_path = sep_dir / 'instr.wav'
        
        cmd = [
            'python', 'scripts/separate.py',
            '--input', input_path,
            '--out_dir', str(sep_dir),
            '--model', args.separation_model
        ]
        
        if not run_command(cmd, 'Source separation'):
            return False
    
    # Step 2: Preprocessing
    logger.info('Step 2/6: Preprocessing')
    preprocessed_path = pre_dir / 'vocal_pre.wav'
    
    cmd = [
        'python', 'scripts/preprocess.py',
        '--vocal', str(vocal_path),
        '--output', str(preprocessed_path),
        '--sr', str(args.sr)
    ]
    
    if args.denoise:
        cmd.append('--denoise')
    if args.vad:
        cmd.append('--vad')
    if args.align:
        cmd.extend(['--align', '--backing', str(backing_path)])
    
    if not run_command(cmd, 'Preprocessing'):
        return False
    
    # Step 3: Extract F0
    logger.info('Step 3/6: Pitch extraction')
    f0_path = f0_dir / 'extracted.npz'
    
    cmd = [
        'python', 'scripts/extract_pitch.py',
        '--input', str(preprocessed_path),
        '--output', str(f0_path),
        '--sr', str(args.sr),
        '--method', args.pitch_method
    ]
    
    if not run_command(cmd, 'Pitch extraction'):
        return False
    
    # Step 4: Infer target F0
    logger.info('Step 4/6: Target pitch inference')
    target_path = target_dir / 'target.npz'
    
    cmd = [
        'python', 'scripts/infer_target_pitch.py',
        '--f0_npz', str(f0_path),
        '--output', str(target_path),
        '--mode', 'heuristic',
        '--root_midi', str(args.root_midi),
        '--scale', args.scale,
        '--vibrato_preserve', str(args.vibrato_preserve)
    ]
    
    if not run_command(cmd, 'Target pitch inference'):
        return False
    
    # Step 5: Correct pitch
    logger.info('Step 5/6: Pitch correction')
    corrected_path = corrected_dir / 'vocal_corrected.wav'
    
    cmd = [
        'python', 'scripts/correct_pitch.py',
        '--vocal', str(preprocessed_path),
        '--target_npz', str(target_path),
        '--output', str(corrected_path),
        '--sr', str(args.sr),
        '--method', args.vocoder_method
    ]
    
    if not run_command(cmd, 'Pitch correction'):
        return False
    
    # Step 6: Post-process and mix
    logger.info('Step 6/6: Post-processing and mixing')
    
    cmd = [
        'python', 'scripts/postprocess.py',
        '--vocal', str(corrected_path),
        '--backing', str(backing_path),
        '--output', output_path,
        '--sr', str(args.sr),
        '--vocal_gain_db', str(args.vocal_gain_db),
        '--target_lufs', str(args.target_lufs)
    ]
    
    if args.deess:
        cmd.append('--deess')
    
    if not run_command(cmd, 'Post-processing and mixing'):
        return False
    
    logger.info(f'=== Pipeline complete! Output: {output_path} ===')
    return True


def run_model_pipeline(input_path, output_path, work_dir, args):
    """
    Run the model-based pipeline (requires trained model).
    Similar to fast pipeline but uses ML model for pitch inference.
    """
    logger.warning('Model-based pipeline not yet fully implemented')
    logger.info('Using heuristic pipeline instead')
    return run_fast_pipeline(input_path, output_path, work_dir, args)


def main():
    parser = argparse.ArgumentParser(
        description='Run autotune pipeline on audio file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output
    parser.add_argument('--input', '-i', help='Input audio file (for separation mode)')
    parser.add_argument('--vocal', '-v', help='Pre-separated vocal file')
    parser.add_argument('--backing', '-b', help='Pre-separated backing file')
    parser.add_argument('--output', '-o', required=True, help='Output file path')
    parser.add_argument('--work_dir', '-w', default='./work',
                       help='Working directory for intermediate files')
    
    # Pipeline mode
    parser.add_argument('--mode', choices=['fast', 'model'], default='fast',
                       help='Pipeline mode: fast (heuristic) or model (ML-based)')
    
    # Audio parameters
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    
    # Separation parameters
    parser.add_argument('--separation_model', default='htdemucs',
                       choices=['htdemucs', 'htdemucs_ft', 'mdx_extra'],
                       help='Demucs model for separation')
    
    # Preprocessing parameters
    parser.add_argument('--denoise', action='store_true', help='Apply denoising')
    parser.add_argument('--vad', action='store_true', help='Apply voice activity detection')
    parser.add_argument('--align', action='store_true', help='Align vocal to backing')
    
    # Pitch extraction parameters
    parser.add_argument('--pitch_method', choices=['crepe', 'librosa'], default='crepe',
                       help='Pitch extraction method')
    
    # Target pitch parameters
    parser.add_argument('--root_midi', type=int, default=60,
                       help='Root note (MIDI number, 60=C4)')
    parser.add_argument('--scale', choices=['major', 'minor', 'chromatic'], default='major',
                       help='Musical scale')
    parser.add_argument('--vibrato_preserve', type=float, default=0.25,
                       help='Vibrato preservation (0-1)')
    
    # Correction parameters
    parser.add_argument('--vocoder_method', choices=['world', 'psola'], default='world',
                       help='Vocoder method')
    
    # Post-processing parameters
    parser.add_argument('--vocal_gain_db', type=float, default=0.0,
                       help='Vocal gain in dB')
    parser.add_argument('--target_lufs', type=float, default=-14.0,
                       help='Target loudness (LUFS)')
    parser.add_argument('--deess', action='store_true', help='Apply de-essing')
    
    # Model parameters (for model mode)
    parser.add_argument('--model_ckpt', help='Model checkpoint path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not (args.vocal and args.backing):
        parser.error('Either --input or both --vocal and --backing must be provided')
    
    if args.input and (args.vocal or args.backing):
        parser.error('Cannot specify both --input and --vocal/--backing')
    
    # Set input path
    input_path = args.input or args.vocal
    
    # Check input exists
    if not os.path.exists(input_path):
        logger.error(f'Input file not found: {input_path}')
        sys.exit(1)
    
    # Create work directory
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        if args.mode == 'fast':
            success = run_fast_pipeline(input_path, args.output, args.work_dir, args)
        else:
            success = run_model_pipeline(input_path, args.output, args.work_dir, args)
        
        if success:
            logger.info('Pipeline completed successfully!')
            sys.exit(0)
        else:
            logger.error('Pipeline failed')
            sys.exit(1)
            
    except Exception as e:
        logger.error(f'Pipeline error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
