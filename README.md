# SUB ai - Small Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Overview
SUB ai is a comprehensive small language model with dual capabilities:
1. **Number Detection**: Detect and recognize digits (0-9) from images
2. **Conversational AI**: Natural text-based conversations using real datasets from Hugging Face

## 🌟 Features

### Number Detection Module
- ✅ Detect numbers from images (digits 0-9)
- ✅ Classify images as "number" or "not a number"
- ✅ CNN-based neural network for high accuracy
- ✅ Trained on MNIST dataset (98-99% accuracy)

### Chat AI Module
- ✅ Text-based conversations trained on 5,000+ real dialogues
- ✅ Multiple Hugging Face datasets (DailyDialog, Empathetic Dialogues)
- ✅ Natural language understanding
- ✅ Sequence-to-sequence neural architecture
- ✅ Rule-based fallback for reliability
- ✅ Interactive chat interface

## 📦 Project Structure
```
SUB-ai/
├── README.md
├── LICENSE               # MIT License
├── CONTRIBUTING.md       # Contribution guidelines
├── DATASETS.md           # Dataset documentation
├── WORKFLOWS.md          # GitHub Actions guide
├── requirements.txt
├── sub_ai.py              # Unified AI interface
├── number_detector.py    # Number detection module
├── chat_ai.py            # Chat AI module
├── train.py              # Number detection training
├── train_chat.py         # Chat model training
├── test_detector.py      # Testing script
├── .github/workflows/    # GitHub Actions
├── models/               # Saved models
└── data/                 # Training data
```

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/subhobhai943/SUB-ai.git
cd SUB-ai

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Training the Models

### Option 1: GitHub Actions (Recommended)

Train on GitHub's servers - no local setup required!

#### Train Number Detection Model
1. Go to [Actions tab](https://github.com/subhobhai943/SUB-ai/actions)
2. Select "Train SUB ai Model"
3. Click "Run workflow"
4. Wait ~5-8 minutes
5. Model automatically commits to repository

#### Train Chat Model
1. Go to [Actions tab](https://github.com/subhobhai943/SUB-ai/actions)
2. Select "Train SUB ai Chat Model"
3. Click "Run workflow"
4. Choose dataset: `daily_dialog` (recommended) or `empathetic_dialogues`
5. Set max samples: `5000` (default)
6. Wait ~8-12 minutes
7. Model automatically commits to repository

**See [WORKFLOWS.md](WORKFLOWS.md) for detailed instructions.**

### Option 2: Local Training

```bash
# Train number detection model
python train.py

# Train chat model with Hugging Face dataset
python train_chat.py
```

**See [DATASETS.md](DATASETS.md) for dataset options and configuration.**

## 💬 Usage

### Unified Interface (Recommended)

Run SUB ai with both capabilities:

```bash
python sub_ai.py
```

Example session:
```
You: Hello!
SUB ai: Hello! How can I help you today?

You: What can you do?
SUB ai: I can detect numbers from images and chat with you!

You: detect test_images/number_5.png
SUB ai: This is a number image! Detected: 5
  Digit: 5
  Confidence: 99.87%

You: Thanks!
SUB ai: You're welcome! Happy to help!
```

### Chat Only

```bash
python chat_ai.py
```

### Number Detection Only

```python
from number_detector import NumberDetector

detector = NumberDetector(model_path='models/sub_ai_model_latest.h5')
result = detector.detect('path/to/image.jpg')

print(result['message'])
print(f"Confidence: {result['confidence']:.2%}")
```

### Python API

```python
from sub_ai import SUBai

# Initialize
ai = SUBai()

# Chat
response = ai.chat("Hello!")
print(response['response'])

# Detect numbers
result = ai.detect_number('image.png')
print(result['message'])
```

## 🧠 Model Architectures

### Number Detection CNN
```
Conv2D (32, 3x3) → ReLU → MaxPooling
           ↓
Conv2D (64, 3x3) → ReLU → MaxPooling
           ↓
      Flatten → Dropout (0.5)
           ↓
   Dense (128) → ReLU → Dropout (0.3)
           ↓
 Dense (10, softmax) → Output (0-9)
```

### Chat Model (Seq2Seq)
```
Embedding (128 dim)
           ↓
Bi-LSTM (256 units) → Dropout (0.3)
           ↓
Bi-LSTM (128 units) → Dropout (0.3)
           ↓
  Dense (256, ReLU)
           ↓
Dense (vocab_size, softmax) → Response
```

## 📈 Performance

| Model | Metric | Value |
|-------|--------|-------|
| Number Detection | Test Accuracy | 98-99% |
| Number Detection | Training Time | 5-8 min |
| Number Detection | Model Size | ~1.5 MB |
| Chat AI (Local Data) | Training Time | 3-5 min |
| Chat AI (HF Dataset) | Training Time | 8-12 min |
| Chat AI | Model Size | ~2 MB |
| Chat AI | Response Time | <50ms |
| Chat AI | Training Samples | 5,000+ |

## 🛣️ Roadmap

### Stage 1 ✅ (Completed)
- [x] Number detection from images
- [x] CNN model training on MNIST
- [x] GitHub Actions workflow
- [x] 98%+ accuracy

### Stage 2 ✅ (Completed)
- [x] Conversational AI chat
- [x] Hugging Face dataset integration
- [x] 5,000+ training samples
- [x] Multiple dataset options

### Stage 3 (In Progress)
- [ ] Improve chat response quality
- [ ] Add context memory
- [ ] Multi-turn conversation support
- [ ] Web interface (Flask/Gradio)

### Stage 4 (Future)
- [ ] Multi-digit number recognition
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Mobile app
- [ ] API endpoints

## 📚 Documentation

- **[README.md](README.md)** - Project overview (this file)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[DATASETS.md](DATASETS.md)** - Available datasets and training guide
- **[WORKFLOWS.md](WORKFLOWS.md)** - GitHub Actions workflows
- **[LICENSE](LICENSE)** - MIT License

## 🤖 Available Datasets

### For Chat Training

1. **DailyDialog** (Default) 🌟
   - 13,000+ natural daily conversations
   - Best for general-purpose chat

2. **Empathetic Dialogues**
   - 25,000+ emotion-aware conversations
   - Great for empathetic AI

3. **Local Dataset** (Fallback)
   - 130+ built-in conversation pairs
   - No internet required

**See [DATASETS.md](DATASETS.md) for detailed information.**

## 🛠️ Technologies

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **Hugging Face Datasets** - Real conversation data
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **LSTM/GRU** - Sequence modeling
- **Matplotlib** - Visualization

## 🐛 Troubleshooting

### Models Not Found
```bash
# Train both models
python train.py          # Number detection
python train_chat.py     # Chat AI
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Dataset Download Issues
```bash
# Use local fallback data
export USE_HF_DATASET=false
python train_chat.py
```

### Low Accuracy
- Increase training epochs
- Use more training data (`MAX_SAMPLES=10000`)
- Try different datasets

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- 🐛 Report bugs
- ✨ Suggest features
- 📝 Improve documentation
- 💻 Write code
- ✅ Add tests
- 🌍 Translate

### Quick Start

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open a Pull Request

**Read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

### Current Priorities

- [ ] Add unit tests
- [ ] Create web interface
- [ ] Improve chat quality
- [ ] Add more datasets
- [ ] Multi-language support
- [ ] Context memory

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

✅ Commercial use  
✅ Modification  
✅ Distribution  
✅ Private use  

## ✍️ Author

**Subhobhai** - [@subhobhai943](https://github.com/subhobhai943)
- Portfolio: [subhadip-portofolio.netlify.app](https://subhadip-portofolio.netlify.app)
- Email: sarkarsubhadip604@gmail.com
- Building AI projects and experimenting with ML

## 🙏 Acknowledgments

- **MNIST Dataset** - Yann LeCun and Corinna Cortes
- **DailyDialog Dataset** - Li et al.
- **Empathetic Dialogues** - Facebook AI Research
- **TensorFlow Team** - Amazing deep learning framework
- **Hugging Face** - Datasets library and platform
- **Open Source Community** - For inspiration and support

## ⭐ Show Your Support

Give a ⭐ if this project helped you!

## 💬 Community

- **Issues**: [Report bugs or request features](https://github.com/subhobhai943/SUB-ai/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/subhobhai943/SUB-ai/discussions)
- **Pull Requests**: [Contribute code](https://github.com/subhobhai943/SUB-ai/pulls)

---

**Status**: Stage 2 Complete ✅ | Hugging Face Integration Added 🤗 | Ready for Training 🚀

**Try it now**: `python sub_ai.py` 💬
