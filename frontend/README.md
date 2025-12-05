# Autotune-AI Web Frontend

A modern, user-friendly web interface for the Autotune-AI audio pitch correction system.

## Features

- 🎨 **Modern UI**: Clean, responsive design with dark theme
- 📤 **Dual Input Modes**: Upload single mixed files or pre-separated stems
- ⚙️ **Full Configuration**: Access all pipeline parameters through the UI
- 📊 **Real-time Progress**: Live progress tracking during processing
- ⬇️ **Direct Download**: Download processed audio directly from the browser
- 📱 **Mobile Friendly**: Responsive design works on all devices

## Quick Start

### Prerequisites

- Python 3.12+
- All dependencies from the main `requirements.txt`
- Flask and werkzeug (included in requirements.txt)

### Installation

1. Install all dependencies:
```bash
cd /path/to/autotune-ai
pip install -r requirements.txt
```

2. Start the web server:
```bash
cd frontend
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

### Docker

You can also run the frontend in Docker:

```bash
# Build the Docker image
docker build -t autotune-ai:latest -f docker/Dockerfile .

# Run with web server exposed
docker run --gpus all -p 5000:5000 -it autotune-ai:latest python frontend/app.py
```

Then access the web interface at `http://localhost:5000`.

## Usage

### Basic Workflow

1. **Upload Audio**
   - Choose "Single Mixed File" for automatic vocal separation
   - Or choose "Pre-Separated Stems" if you have vocal and backing tracks

2. **Configure Settings**
   - **Mode**: Fast (heuristic) or Model (ML-based)
   - **Root Note**: Select the key of your song
   - **Scale**: Major, Minor, or Chromatic
   - **Vibrato Preservation**: Control natural-sounding results (0-100%)

3. **Advanced Settings** (Optional)
   - Separation model selection
   - Pitch detection method
   - Vocoder type
   - Preprocessing options

4. **Process**
   - Click "Process Audio" to start
   - Monitor real-time progress
   - Download your corrected audio when complete

### Configuration Options

#### Basic Settings

- **Mode**
  - `fast`: Uses heuristic scale quantization (faster)
  - `model`: Uses ML model for prediction (requires trained model)

- **Root Note**: MIDI note number (60 = C4, 69 = A4 440Hz)

- **Scale**
  - `major`: Major scale correction
  - `minor`: Minor scale correction
  - `chromatic`: All 12 semitones

- **Vibrato Preservation**: 0-1 (0% = robotic, 100% = natural)

#### Advanced Settings

- **Separation Model**
  - `htdemucs`: Standard model (good balance)
  - `htdemucs_ft`: Fine-tuned model (better quality)
  - `mdx_extra`: Alternative architecture

- **Pitch Detection**
  - `crepe`: Deep learning-based (more accurate, GPU recommended)
  - `librosa`: Traditional DSP (faster, CPU-friendly)

- **Vocoder**
  - `world`: High-quality WORLD vocoder (recommended)
  - `psola`: Faster PSOLA method

- **Sample Rate**: 22050, 44100, or 48000 Hz

- **Preprocessing Options**
  - Noise Reduction: Remove background noise
  - VAD: Voice Activity Detection
  - Align: Align vocal to backing track timing

## API Endpoints

The frontend exposes a REST API that can be used programmatically:

### POST `/upload`
Upload and process audio files.

**Form Data:**
- `audioFile`: Audio file (for single mode)
- `vocalFile`: Vocal track (for separated mode)
- `backingFile`: Backing track (for separated mode)
- All configuration parameters as form fields

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Processing started"
}
```

### GET `/status/<job_id>`
Check processing status.

**Response:**
```json
{
  "status": "processing",
  "progress": 50,
  "created_at": "2024-12-04T...",
  "params": {...}
}
```

Status values: `queued`, `processing`, `completed`, `failed`

### GET `/download/<job_id>`
Download processed audio file.

Returns the WAV file as an attachment.

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "autotune-ai",
  "version": "1.0.0"
}
```

## Environment Variables

Configure the application using environment variables:

- `PORT`: Server port (default: 5000)
- `DEBUG`: Enable debug mode (default: false)
- `SECRET_KEY`: Flask secret key (auto-generated if not set)

Example:
```bash
export PORT=8080
export DEBUG=true
python app.py
```

## Production Deployment

### Using Gunicorn

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using systemd

Create a systemd service file `/etc/systemd/system/autotune-ai.service`:

```ini
[Unit]
Description=Autotune-AI Web Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/autotune-ai/frontend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable autotune-ai
sudo systemctl start autotune-ai
```

### Using Nginx

Proxy the Flask app with Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Increase timeout for long processing
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Increase max body size for large files
    client_max_body_size 500M;
}
```

## File Management

Uploaded and processed files are stored in:
- `frontend/uploads/`: Uploaded files
- `frontend/outputs/`: Processed files

**Note**: These directories can grow large. Consider implementing:
- Automatic cleanup of old files
- Cloud storage integration (S3, etc.)
- Database for job tracking

## Troubleshooting

### Files Not Processing

1. Check that all required dependencies are installed
2. Verify GPU availability if using CREPE or model mode
3. Check server logs for errors
4. Ensure sufficient disk space

### Slow Processing

1. Use `librosa` instead of `crepe` for faster pitch detection
2. Use `psola` instead of `world` vocoder
3. Reduce sample rate to 22050 Hz
4. Enable GPU acceleration

### File Upload Errors

1. Check file size limit (500MB default)
2. Verify file format is supported
3. Ensure sufficient disk space

## Security Considerations

For production deployments:

1. **Set a strong SECRET_KEY**:
   ```bash
   export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

2. **Use HTTPS**: Always use SSL/TLS in production

3. **Implement authentication**: Add user authentication for access control

4. **Rate limiting**: Implement rate limiting to prevent abuse

5. **File validation**: Validate uploaded files thoroughly

6. **Sanitize inputs**: All user inputs are sanitized using `secure_filename`

## License

This frontend is part of the Autotune-AI project and follows the same license.

## Support

For issues or questions:
- GitHub Issues: https://github.com/groxaxo/autotune-ai/issues
- Main Documentation: See parent README.md
