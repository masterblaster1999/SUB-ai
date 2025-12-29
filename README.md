# SUB ai - Small Language Model for Number Detection

## Overview
SUB ai is a small language model designed to detect and recognize numbers from images. In its first stage, it focuses on identifying numerical digits and distinguishing number images from non-number images.

## Features
- ✅ Detect numbers from images
- ✅ Classify images as "number" or "not a number"
- 🚀 Lightweight and efficient
- 📊 Built with modern ML frameworks

## Project Structure
```
SUB-ai/
├── README.md
├── requirements.txt
├── number_detector.py    # Main detection model
├── train.py             # Training script (coming soon)
└── models/              # Saved models directory
```

## Installation

```bash
# Clone the repository
git clone https://github.com/subhobhai943/SUB-ai.git
cd SUB-ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from number_detector import NumberDetector

# Initialize the detector
detector = NumberDetector()

# Detect numbers from an image
result = detector.detect('path/to/image.jpg')
print(result)
```

## Roadmap

### Stage 1 (Current)
- [x] Project setup
- [ ] Implement basic number detection
- [ ] Train initial model
- [ ] Add validation

### Stage 2 (Upcoming)
- [ ] Improve accuracy
- [ ] Add multi-digit recognition
- [ ] Web interface

## Technologies
- Python 3.8+
- TensorFlow/Keras or PyTorch
- OpenCV for image processing
- NumPy for numerical operations

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
MIT License

## Author
Subhobhai - [@subhobhai943](https://github.com/subhobhai943)
