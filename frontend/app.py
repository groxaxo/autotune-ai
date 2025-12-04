"""
Autotune-AI Web Frontend
Flask-based web interface for audio pitch correction
"""
import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import subprocess
import uuid
import json
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'autotune-ai-secret-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['OUTPUT_FOLDER'] = Path(__file__).parent / 'outputs'

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(exist_ok=True)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Store job status
jobs_status = {}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_job_id():
    """Generate a unique job ID."""
    return str(uuid.uuid4())


def run_pipeline_async(job_id, input_path, output_path, params):
    """
    Run the autotune pipeline asynchronously.
    
    Args:
        job_id: Unique job identifier
        input_path: Path to input audio file
        output_path: Path for output file
        params: Dictionary of pipeline parameters
    """
    try:
        jobs_status[job_id]['status'] = 'processing'
        jobs_status[job_id]['progress'] = 10
        
        # Build command
        script_path = Path(__file__).parent.parent / 'scripts' / 'run_pipeline.py'
        cmd = ['python', str(script_path)]
        
        # Add input/output
        if params.get('vocal_path') and params.get('backing_path'):
            cmd.extend(['--vocal', params['vocal_path']])
            cmd.extend(['--backing', params['backing_path']])
        else:
            cmd.extend(['--input', str(input_path)])
        
        cmd.extend(['--output', str(output_path)])
        
        # Add parameters
        cmd.extend(['--mode', params.get('mode', 'fast')])
        cmd.extend(['--root_midi', str(params.get('root_midi', 60))])
        cmd.extend(['--scale', params.get('scale', 'major')])
        cmd.extend(['--vibrato_preserve', str(params.get('vibrato_preserve', 0.25))])
        cmd.extend(['--separation_model', params.get('separation_model', 'htdemucs')])
        cmd.extend(['--pitch_method', params.get('pitch_method', 'crepe')])
        cmd.extend(['--vocoder_method', params.get('vocoder_method', 'world')])
        cmd.extend(['--sr', str(params.get('sample_rate', 44100))])
        
        if params.get('denoise'):
            cmd.append('--denoise')
        if params.get('vad'):
            cmd.append('--vad')
        if params.get('align'):
            cmd.append('--align')
        
        logger.info(f'Running command: {" ".join(cmd)}')
        
        # Run the pipeline
        jobs_status[job_id]['progress'] = 30
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            jobs_status[job_id]['status'] = 'completed'
            jobs_status[job_id]['progress'] = 100
            jobs_status[job_id]['output_path'] = str(output_path)
            logger.info(f'Job {job_id} completed successfully')
        else:
            jobs_status[job_id]['status'] = 'failed'
            jobs_status[job_id]['error'] = result.stderr
            logger.error(f'Job {job_id} failed: {result.stderr}')
    
    except Exception as e:
        jobs_status[job_id]['status'] = 'failed'
        jobs_status[job_id]['error'] = str(e)
        logger.error(f'Job {job_id} exception: {e}', exc_info=True)


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing."""
    try:
        # Check if files are present
        if 'audioFile' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audioFile']
        vocal_file = request.files.get('vocalFile')
        backing_file = request.files.get('backingFile')
        
        if audio_file.filename == '' and (not vocal_file or vocal_file.filename == ''):
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate job ID
        job_id = generate_job_id()
        
        # Get parameters from request
        params = {
            'mode': request.form.get('mode', 'fast'),
            'root_midi': int(request.form.get('rootNote', 60)),
            'scale': request.form.get('scale', 'major'),
            'vibrato_preserve': float(request.form.get('vibratoPreserve', 0.25)),
            'separation_model': request.form.get('separationModel', 'htdemucs'),
            'pitch_method': request.form.get('pitchMethod', 'crepe'),
            'vocoder_method': request.form.get('vocoderMethod', 'world'),
            'sample_rate': int(request.form.get('sampleRate', 44100)),
            'denoise': request.form.get('denoise') == 'true',
            'vad': request.form.get('vad') == 'true',
            'align': request.form.get('align') == 'true',
        }
        
        # Save files
        input_path = None
        output_filename = None
        
        if vocal_file and vocal_file.filename and backing_file and backing_file.filename:
            # Pre-separated stems provided
            vocal_filename = secure_filename(vocal_file.filename)
            backing_filename = secure_filename(backing_file.filename)
            
            vocal_path = app.config['UPLOAD_FOLDER'] / f"{job_id}_vocal_{vocal_filename}"
            backing_path = app.config['UPLOAD_FOLDER'] / f"{job_id}_backing_{backing_filename}"
            
            vocal_file.save(vocal_path)
            backing_file.save(backing_path)
            
            params['vocal_path'] = str(vocal_path)
            params['backing_path'] = str(backing_path)
            
            output_filename = f"{job_id}_corrected.wav"
        else:
            # Single mixed file
            filename = secure_filename(audio_file.filename)
            input_path = app.config['UPLOAD_FOLDER'] / f"{job_id}_{filename}"
            audio_file.save(input_path)
            output_filename = f"{job_id}_corrected.wav"
        
        output_path = app.config['OUTPUT_FOLDER'] / output_filename
        
        # Initialize job status
        jobs_status[job_id] = {
            'status': 'queued',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'params': params
        }
        
        # Start processing in background (in production, use Celery or similar)
        import threading
        thread = threading.Thread(
            target=run_pipeline_async,
            args=(job_id, input_path, output_path, params)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Processing started'
        })
    
    except Exception as e:
        logger.error(f'Upload error: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status."""
    if job_id not in jobs_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job_info = jobs_status[job_id].copy()
    
    # Add download URL if completed
    if job_info['status'] == 'completed' and 'output_path' in job_info:
        job_info['download_url'] = url_for('download_file', job_id=job_id)
    
    return jsonify(job_info)


@app.route('/download/<job_id>')
def download_file(job_id):
    """Download processed audio file."""
    if job_id not in jobs_status:
        return jsonify({'error': 'Job not found'}), 404
    
    job_info = jobs_status[job_id]
    
    if job_info['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    output_path = Path(job_info['output_path'])
    
    if not output_path.exists():
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f'autotune_corrected_{job_id[:8]}.wav'
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'autotune-ai',
        'version': '1.0.0'
    })


if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f'Starting Autotune-AI web server on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)
