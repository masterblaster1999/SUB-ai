# Contributing to SUB ai

Thank you for considering contributing to SUB ai! We welcome contributions from everyone. This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, gender identity, sexual orientation, disability, personal appearance, race, ethnicity, age, religion, or nationality.

### Expected Behavior

- Be respectful and considerate
- Welcome newcomers and help them get started
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or trolling
- Offensive comments or personal attacks
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

## How Can I Contribute?

### 1. Report Bugs 🐛

Before creating a bug report:
- Check the [existing issues](https://github.com/subhobhai943/SUB-ai/issues) to avoid duplicates
- Gather information about the bug
- Test with the latest version

**Good Bug Report Includes:**
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, etc.)
- Error messages and logs

**Example:**
```markdown
**Bug**: Chat model fails to load vocabulary

**Steps to Reproduce:**
1. Run `python chat_ai.py`
2. Model loads but vocabulary doesn't

**Expected:** Both model and vocab load successfully
**Actual:** Error: "Vocabulary file not found"

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10
- TensorFlow: 2.13.0
```

### 2. Suggest Features ✨

We love new ideas! Before suggesting:
- Check if it's already been suggested
- Ensure it aligns with project goals
- Consider if it benefits most users

**Good Feature Request Includes:**
- Clear description of the feature
- Use cases and benefits
- Possible implementation approach
- Examples from similar projects (optional)

### 3. Improve Documentation 📚

Documentation improvements are always welcome:
- Fix typos or grammatical errors
- Clarify confusing sections
- Add examples and tutorials
- Translate documentation
- Update outdated information

### 4. Contribute Code 💻

Areas where you can help:

**Model Improvements:**
- Better neural architectures
- Training optimizations
- New model types
- Performance enhancements

**Dataset Integration:**
- Add new Hugging Face datasets
- Improve data preprocessing
- Create custom datasets
- Data augmentation techniques

**Features:**
- Web interface (Flask/FastAPI)
- Voice input/output
- Multi-language support
- Context memory
- API endpoints

**Testing:**
- Unit tests
- Integration tests
- Test coverage improvements
- Performance benchmarks

**Infrastructure:**
- CI/CD improvements
- Docker containerization
- Deployment scripts
- Cloud integration

## Getting Started

### 1. Fork the Repository

Click the "Fork" button at the top right of the [SUB ai repository](https://github.com/subhobhai943/SUB-ai).

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/SUB-ai.git
cd SUB-ai
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch Naming Convention:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

## Development Workflow

### 1. Make Your Changes

Write clean, well-documented code following the style guidelines below.

### 2. Test Your Changes

```bash
# Run existing tests
pytest

# Test specific functionality
python test_detector.py
python chat_ai.py

# For chat model changes
python train_chat.py  # Quick test with local data
```

### 3. Format Code

```bash
# Auto-format with black
black *.py

# Check linting
flake8 *.py
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: multi-language support for chat"
```

**Good Commit Messages:**
- Use present tense ("Add feature" not "Added feature")
- Be specific and descriptive
- Reference issue numbers when applicable

**Examples:**
```
✅ Add empathetic dialogues dataset support
✅ Fix tokenizer initialization bug in train_chat.py
✅ Update README with installation instructions
✅ Refactor number_detector.py for better performance
❌ Added stuff
❌ Fixed bug
❌ Update
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

## Pull Request Process

### 1. Create Pull Request

Go to your fork on GitHub and click "New Pull Request".

### 2. Fill Out PR Template

**Good PR Description Includes:**
- Summary of changes
- Motivation and context
- Related issue numbers
- Testing done
- Screenshots (for UI changes)
- Checklist completion

**Example:**
```markdown
## Description
Adds support for Empathetic Dialogues dataset from Hugging Face.

## Motivation
Improves chat quality by training on emotion-aware conversations.

## Changes
- Added `load_empathetic_dataset()` function
- Updated workflow to support dataset selection
- Added documentation in DATASETS.md

## Testing
- [x] Tested locally with 1000 samples
- [x] Verified GitHub Actions workflow
- [x] Confirmed fallback to local data works

## Related Issues
Closes #3

## Screenshots
[Training output showing 5000 samples loaded]
```

### 3. Code Review

- Be patient and respectful during review
- Address all feedback
- Make requested changes promptly
- Ask questions if something is unclear

### 4. Merge

Once approved, a maintainer will merge your PR. Congratulations! 🎉

## Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use 4 spaces for indentation (not tabs)
def train_model(epochs=100, batch_size=32):
    """Train the chat model.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        History: Training history object
    """
    # Clear comments explaining logic
    model = build_model()
    
    # Descriptive variable names
    training_data = load_data()
    
    return model.fit(training_data, epochs=epochs)


# Class naming: PascalCase
class ChatModelTrainer:
    pass

# Function/variable naming: snake_case
def load_dataset():
    pass

# Constants: UPPER_CASE
MAX_SEQUENCE_LENGTH = 50
```

### Documentation Style

```python
def complex_function(param1, param2, param3=None):
    """
    Brief description of what the function does.
    
    More detailed explanation if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2
        param3 (optional, list): Description of param3.
            Defaults to None.
    
    Returns:
        dict: Description of return value
    
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails
    
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['status'])
        'success'
    """
    pass
```

### Commit Message Style

```
Type: Brief summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain the problem this commit solves and why you solved it
this way.

References: #123, #456
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Formatting, missing semicolons, etc.
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance tasks

## Testing Guidelines

### Writing Tests

```python
import pytest
from number_detector import NumberDetector

def test_number_detection():
    """Test basic number detection functionality."""
    detector = NumberDetector()
    result = detector.detect('test_images/number_5.png')
    
    assert result['status'] == 'success'
    assert result['is_number'] == True
    assert result['predicted_digit'] == 5

def test_non_number_detection():
    """Test that non-numbers are correctly identified."""
    detector = NumberDetector()
    result = detector.detect('test_images/random.png')
    
    assert result['is_number'] == False
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_detector.py

# Run specific test
pytest test_detector.py::test_number_detection
```

## Community

### Get Help

- **Issues**: [GitHub Issues](https://github.com/subhobhai943/SUB-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/subhobhai943/SUB-ai/discussions)
- **Email**: sarkarsubhadip604@gmail.com

### Stay Updated

- Watch the repository for updates
- Star the project if you find it useful
- Follow [@subhobhai943](https://github.com/subhobhai943)

### Recognition

All contributors will be:
- Listed in the project README
- Acknowledged in release notes
- Given credit for their work

## Project Structure

```
SUB-ai/
├── .github/
│   └── workflows/          # GitHub Actions
├── models/                # Trained models
├── data/                  # Training data
├── test_images/           # Test images
├── number_detector.py     # Number detection module
├── chat_ai.py             # Chat AI module
├── sub_ai.py              # Unified interface
├── train.py               # Number model training
├── train_chat.py          # Chat model training
├── test_detector.py       # Testing script
├── requirements.txt       # Dependencies
├── README.md              # Project overview
├── DATASETS.md            # Dataset guide
├── WORKFLOWS.md           # GitHub Actions guide
├── CONTRIBUTING.md        # This file
└── LICENSE                # MIT License
```

## Areas Needing Help

Current priorities (as of December 2025):

1. **High Priority**
   - [ ] Add unit tests for all modules
   - [ ] Create web interface (Flask/Gradio)
   - [ ] Improve chat response quality
   - [ ] Add more Hugging Face datasets

2. **Medium Priority**
   - [ ] Multi-language support
   - [ ] Context memory for conversations
   - [ ] Voice input/output
   - [ ] Docker containerization

3. **Low Priority**
   - [ ] Mobile app
   - [ ] API documentation
   - [ ] Performance benchmarks
   - [ ] Cloud deployment guides

## Questions?

If you have questions not covered here:
1. Check existing [issues](https://github.com/subhobhai943/SUB-ai/issues)
2. Open a new issue with the `question` label
3. Reach out via email

---

**Thank you for contributing to SUB ai! 🚀**

Your contributions make this project better for everyone.
