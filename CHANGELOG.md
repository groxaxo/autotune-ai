# Changelog

All notable changes to the Autotune-AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - December 2024

#### Web Frontend (NEW!)
- **Modern Web Interface**: Flask-based web server with real-time progress tracking
  - Responsive HTML/CSS/JS frontend with modern dark theme
  - Dual input modes: single mixed file or pre-separated stems
  - Full parameter configuration through UI
  - Real-time progress updates during processing
  - Direct download of processed audio files
  - Mobile-friendly responsive design

- **REST API**: 
  - `/upload` - Upload and process audio files
  - `/status/<job_id>` - Check processing status
  - `/download/<job_id>` - Download processed audio
  - `/health` - Health check endpoint

#### Deployment & Infrastructure
- **Docker Compose**: One-command deployment with `docker-compose.yml`
  - Web service configuration with GPU support
  - Batch processing service configuration
  - Volume management for uploads and outputs
  - Environment variable configuration

- **Quick Start Script**: `run_frontend.sh` for easy local setup
  - Automatic virtual environment creation
  - Dependency installation
  - GPU detection
  - Server startup

- **Enhanced Dockerfile**:
  - Port 5000 exposed for web interface
  - Frontend directory creation
  - Optimized for both interactive and web modes

#### Documentation
- **INSTALLATION.md**: Comprehensive installation guide
  - Multiple installation methods
  - GPU setup instructions
  - Troubleshooting section
  - Verification steps

- **CONTRIBUTING.md**: Contributor guidelines
  - Development setup
  - Coding standards
  - Testing guidelines
  - Pull request process

- **frontend/README.md**: Web interface documentation
  - Usage instructions
  - API documentation
  - Configuration options
  - Production deployment guide

- **Enhanced README.md**:
  - Improved installation instructions with 4 options
  - Web interface quick start section
  - Better organization and navigation
  - Links to all documentation

#### Configuration & Dependencies
- Added Flask (>=3.0.0) for web server
- Added Werkzeug (>=3.0.0) for WSGI utilities
- Updated .gitignore for frontend directories
- Added .dockerignore for optimized builds

### Changed
- Reorganized README.md for better user experience
- Updated installation section with multiple options
- Enhanced project structure documentation
- Improved quick start instructions

### Security
- ✅ CodeQL security scan: 0 vulnerabilities detected
- File upload sanitization with `secure_filename`
- Input validation on all user inputs
- Environment variable configuration for secrets

## [1.0.0] - 2024

### Initial Release

#### Core Features
- **Six-Stage Audio Processing Pipeline**:
  1. Source Separation (Demucs)
  2. Preprocessing (noise reduction, VAD, alignment)
  3. Pitch Extraction (CREPE, librosa)
  4. Target Pitch Inference (heuristic, ML model)
  5. Pitch Correction (WORLD vocoder, PSOLA)
  6. Post-processing (LUFS normalization, de-essing, mixing)

- **Machine Learning**:
  - CNN + Transformer pitch predictor (3.7M parameters)
  - Training pipeline with validation and checkpointing
  - Hybrid mode combining ML and heuristic approaches
  - GPU acceleration with automatic CPU fallback

- **Batch Processing**:
  - Snakemake workflow orchestration
  - Parallel processing support
  - DAG-based dependency management
  - YAML configuration

#### Infrastructure
- Docker support with CUDA 12.2
- Comprehensive test suite (15 tests, 100% passing)
- CI/CD with GitHub Actions
- Security scanning with CodeQL

#### Documentation
- Detailed README.md
- USAGE.md with examples
- IMPLEMENTATION_SUMMARY.md with architecture details
- Example scripts and demos

## Roadmap

### Planned Features

#### Near Term
- [ ] Pre-trained model checkpoints
- [ ] Automated training data generation
- [ ] Additional musical scales (pentatonic, blues)
- [ ] Audio visualization in web interface
- [ ] User authentication for web interface

#### Medium Term
- [ ] HiFi-GAN vocoder integration
- [ ] Real-time processing mode
- [ ] Automatic key detection
- [ ] Formant preservation controls
- [ ] Batch upload in web interface

#### Long Term
- [ ] VST/AU plugin for DAWs
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment guides
- [ ] Multi-user web interface
- [ ] REST API with authentication

## Migration Guide

### Upgrading from CLI-only to Web Interface

If you were using the CLI, the web interface is now available but completely optional:

**Before (CLI only):**
```bash
python scripts/run_pipeline.py --input audio.wav --output corrected.wav
```

**Now (CLI still works):**
```bash
# CLI - same as before
python scripts/run_pipeline.py --input audio.wav --output corrected.wav

# Web interface - new option
./run_frontend.sh  # or docker-compose up autotune-ai
```

All existing CLI functionality remains unchanged. The web interface is an additional option.

## Contributors

Thank you to all contributors who have helped make Autotune-AI better!

See the [Contributors](https://github.com/groxaxo/autotune-ai/graphs/contributors) page for a full list.

## License

This project is released for educational and research purposes. See [LICENSE](LICENSE) file for details.
