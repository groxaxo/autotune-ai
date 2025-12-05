# Contributing to Autotune-AI

Thank you for your interest in contributing to Autotune-AI! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contribution Workflow](#contribution-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility for mistakes and learn from them

## Getting Started

### Areas for Contribution

We welcome contributions in the following areas:

**Core Features:**
- Additional musical scales and temperaments (pentatonic, blues, harmonic minor)
- Automatic key detection and scale inference
- Formant preservation and manipulation
- Real-time processing with streaming audio

**ML Improvements:**
- Enhanced model architectures (Conformer, WaveNet-style)
- Automated training data generation pipeline
- Few-shot learning for artist-specific models
- Multi-task learning (pitch + timing + dynamics)

**Frontend Enhancements:**
- Authentication and user management
- Audio visualization (waveform, spectrogram)
- Batch upload and processing
- Mobile-responsive improvements

**Integration & Deployment:**
- HiFi-GAN or DiffWave vocoder integration
- VST/AU plugin for DAW integration
- Cloud deployment guides (AWS, GCP, Azure)
- REST API with comprehensive documentation

**Performance:**
- ONNX export for faster inference
- Streaming pipeline for long audio files
- Quantization and optimization for edge devices

**Documentation:**
- Video tutorials
- More examples and use cases
- Troubleshooting guides
- Translation to other languages

## Development Setup

### Prerequisites

- Python 3.12+
- Git
- Ubuntu 24.04 (or compatible Linux)
- NVIDIA GPU (optional, for GPU features)

### Setup Instructions

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/autotune-ai.git
   cd autotune-ai
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/groxaxo/autotune-ai.git
   ```

4. **Create virtual environment**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies**

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8  # Development tools
   ```

6. **Verify setup**

   ```bash
   pytest tests/ -v
   flake8 scripts/ models/ --max-line-length=100
   ```

## Contribution Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Your Changes

- Write clear, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

Detailed explanation of what changed and why.
- Bullet points for specific changes
- Reference issues if applicable (#123)
"
```

Commit message format:
- First line: Brief summary (50 chars max)
- Blank line
- Detailed description
- Reference issues: "Fixes #123" or "Related to #456"

### 4. Keep Your Branch Updated

```bash
git fetch upstream
git rebase upstream/main
```

Resolve any conflicts that arise.

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template
5. Submit the PR

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Single quotes preferred, double for docstrings
- **Imports**: Organized by standard library, third-party, local

### Code Quality Tools

**Linting:**
```bash
flake8 scripts/ models/ frontend/ --max-line-length=100 --exclude=venv,__pycache__
```

**Type Hints** (encouraged but not required):
```python
def process_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Process audio array."""
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function.
    
    Longer description if needed, explaining what the function
    does in more detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
    """
    pass
```

### Code Organization

- Keep functions focused and small (< 50 lines)
- Use meaningful variable names
- Avoid magic numbers (use constants)
- Handle errors gracefully
- Log important operations

Example:
```python
import logging

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
MAX_PITCH_HZ = 1000

def extract_pitch(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract pitch from audio."""
    try:
        logger.info(f'Extracting pitch from audio of length {len(audio)}')
        # Implementation
        return pitch
    except Exception as e:
        logger.error(f'Pitch extraction failed: {e}')
        raise
```

## Testing Guidelines

### Writing Tests

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Test files should be named `test_*.py`
3. **Structure**: Use pytest conventions

Example test:
```python
import pytest
import numpy as np
from scripts.utils import read_wav, write_wav

def test_audio_roundtrip(tmp_path):
    """Test that audio can be written and read back."""
    # Create test data
    sr = 22050
    duration = 1.0
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
    
    # Write and read
    path = tmp_path / 'test.wav'
    write_wav(path, audio, sr)
    audio_read, sr_read = read_wav(path)
    
    # Verify
    assert sr_read == sr
    assert len(audio_read) == len(audio)
    np.testing.assert_allclose(audio_read, audio, rtol=1e-5)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage
pytest tests/ -v --cov=scripts --cov=models --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Coverage

Aim for:
- **Unit tests**: 80%+ coverage for core functions
- **Integration tests**: Major workflows covered
- **Edge cases**: Boundary conditions tested

## Documentation

### Code Documentation

- All public functions must have docstrings
- Complex logic should have inline comments
- README files for each major component

### User Documentation

When adding features:
- Update `README.md` with usage examples
- Update `USAGE.md` with detailed instructions
- Add examples to `examples/` directory
- Update `INSTALLATION.md` if setup changes

### Changelog

Update changelog for significant changes:
```markdown
## [Unreleased]
### Added
- New feature X for doing Y
### Changed
- Modified behavior of Z
### Fixed
- Fixed bug in component A
```

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass
   ```bash
   pytest tests/ -v
   ```

2. **Check code style**: Run linter
   ```bash
   flake8 scripts/ models/ frontend/ --max-line-length=100
   ```

3. **Update documentation**: Add/update relevant docs

4. **Rebase on main**: Ensure clean merge
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### PR Template

When creating a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Tested manually with [describe scenario]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. Automated checks will run (tests, linting)
2. Maintainers will review your code
3. Address feedback in new commits
4. Once approved, it will be merged

### After Merge

1. **Delete your branch**
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Update your fork**
   ```bash
   git checkout main
   git pull upstream main
   ```

## Development Tips

### Debugging

Use logging instead of print statements:
```python
import logging
logger = logging.getLogger(__name__)

logger.debug('Detailed info for debugging')
logger.info('General information')
logger.warning('Warning message')
logger.error('Error occurred')
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile a function
cProfile.run('my_function()', 'output.prof')
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Git Tips

**Amend last commit:**
```bash
git add .
git commit --amend --no-edit
```

**Interactive rebase for clean history:**
```bash
git rebase -i HEAD~3  # Last 3 commits
```

**Stash changes temporarily:**
```bash
git stash
git pull
git stash pop
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/groxaxo/autotune-ai/discussions)
- **Bugs**: Open an [Issue](https://github.com/groxaxo/autotune-ai/issues)
- **Chat**: Join our community (link in README)

## Recognition

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Recognized in the project community

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to Autotune-AI! 🎵
