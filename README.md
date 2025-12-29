# SUB ai - Small Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub Release](https://img.shields.io/github/v/release/subhobhai943/SUB-ai)](https://github.com/subhobhai943/SUB-ai/releases)

## Overview
SUB ai is a comprehensive small language model with dual capabilities:
1. **Number Detection**: Detect and recognize digits (0-9) from images
2. **Conversational AI**: Natural text-based conversations using real datasets from Hugging Face

**Models are distributed via [GitHub Releases](https://github.com/subhobhai943/SUB-ai/releases)** - keeping the repository lightweight!

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

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/subhobhai943/SUB-ai.git
cd SUB-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

**Option A: Automatic Download (Recommended)**

```bash
# Download all models from latest releases
python download_models.py

# Or download specific model
python download_models.py number  # Number detection only
python download_models.py chat    # Chat model only
```

**Option B: Manual Download**

1. Go to [Releases](https://github.com/subhobhai943/SUB-ai/releases)
2. Download latest models:
   - Number Detection: `sub_ai_model_latest.h5`
   - Chat Model: `sub_ai_chat_latest.h5` + `chat_vocab.pkl`
3. Place in `models/` directory

**Option C: Train Your Own**

```bash
python train.py        # Number detection
python train_chat.py   # Chat model
```

### 4. Run SUB ai

```bash
python sub_ai.py
```

## 🎯 Training Models

### GitHub Actions Training (Recommended)

Train on GitHub's cloud infrastructure and get models via releases!

#### Train Number Detection Model

1. Go to [Actions tab](https://github.com/subhobhai943/SUB-ai/actions)
2. Select "Train SUB ai Model"
3. Click "Run workflow"
4. Configure options (epochs, batch size)
5. Wait ~5-8 minutes
6. **Model released automatically!** 🎉
7. Download: `python download_models.py number`

#### Train Chat Model

1. Go to [Actions tab](https://github.com/subhobhai943/SUB-ai/actions)
2. Select "Train SUB ai Chat Model"
3. Click "Run workflow"
4. Choose dataset: `daily_dialog` (recommended) or `empathetic_dialogues`
5. Set max samples: `5000` (more = better)
6. Wait ~8-12 minutes
7. **Model released automatically!** 🎉
8. Download: `python download_models.py chat`

**See [WORKFLOWS.md](WORKFLOWS.md) for detailed instructions.**

### Local Training

```bash
# Train number detection
python train.py

# Train chat model with Hugging Face dataset
python train_chat.py

# Train chat with specific dataset
export HF_DATASET=empathetic_dialogues
export MAX_SAMPLES=10000
python train_chat.py
```

**See [DATASETS.md](DATASETS.md) for dataset options.**

## 💬 Usage Examples

### Unified Interface

```bash
python sub_ai.py
```

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

### Number Detection API

```python
from number_detector import NumberDetector

detector = NumberDetector(model_path='models/sub_ai_model_latest.h5')
result = detector.detect('path/to/image.jpg')

print(result['message'])
print(f"Confidence: {result['confidence']:.2%}")
if result.get('predicted_digit') is not None:
    print(f"Detected Digit: {result['predicted_digit']}")
```

## 📦 Model Distribution

### Why GitHub Releases?

✅ **Lightweight Repository** - No large binary files in git  
✅ **Version Control** - Each training run creates a versioned release  
✅ **Easy Downloads** - Direct download links for models  
✅ **Automatic Updates** - New releases created after each training  
✅ **Artifact Storage** - Models also stored as GitHub Actions artifacts (90 days)

### Accessing Models

**Latest Models**: [View Releases](https://github.com/subhobhai943/SUB-ai/releases)

**Download Script**:
```bash
# All models
python download_models.py

# Specific model
python download_models.py number
python download_models.py chat
```

**Manual Download**:
1. Visit [Releases page](https://github.com/subhobhai943/SUB-ai/releases)
2. Find latest release for your model type
3. Download model files
4. Place in `models/` directory

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
  Dense (256, ReLU) → Dropout (0.2)
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
- [x] GitHub Actions workflows
- [x] 98%+ accuracy

### Stage 2 ✅ (Completed)
- [x] Conversational AI chat
- [x] Hugging Face dataset integration
- [x] 5,000+ training samples
- [x] Multiple dataset options
- [x] GitHub Releases for model distribution

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
- [ ] REST API

## 📚 Documentation

- **[README.md](README.md)** - Project overview (this file)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[DATASETS.md](DATASETS.md)** - Dataset guide
- **[WORKFLOWS.md](WORKFLOWS.md)** - GitHub Actions guide
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
- **LSTM/Bi-LSTM** - Sequence modeling
- **Matplotlib** - Visualization

## 🐛 Troubleshooting

### Models Not Found

```bash
# Download from releases
python download_models.py

# Or train locally
python train.py          # Number detection
python train_chat.py     # Chat AI
```

### Download Errors

```bash
# Check releases page
# https://github.com/subhobhai943/SUB-ai/releases

# Or train models locally instead
python train.py
python train_chat.py
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

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Guide

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make changes and commit
4. Push and open a Pull Request

### Priority Areas

- [ ] Add unit tests
- [ ] Create web interface
- [ ] Improve chat quality
- [ ] Add more datasets
- [ ] Multi-language support
- [ ] Context memory

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

✅ Commercial use | ✅ Modification | ✅ Distribution | ✅ Private use

## ✍️ Author

**Subhobhai** - [@subhobhai943](https://github.com/subhobhai943)
- Portfolio: [subhadip-portofolio.netlify.app](https://subhadip-portofolio.netlify.app)
- Email: sarkarsubhadip604@gmail.com

## 🙏 Acknowledgments

- **MNIST Dataset** - Yann LeCun and Corinna Cortes
- **DailyDialog Dataset** - Li et al.
- **Empathetic Dialogues** - Facebook AI Research  
- **TensorFlow Team** - Deep learning framework
- **Hugging Face** - Datasets library and platform
- **Open Source Community** - Inspiration and support

## ⭐ Support

Give a ⭐ if this project helped you!

## 💬 Community

- **Issues**: [Report bugs or request features](https://github.com/subhobhai943/SUB-ai/issues)
- **Discussions**: [Ask questions](https://github.com/subhobhai943/SUB-ai/discussions)
- **Releases**: [Download models](https://github.com/subhobhai943/SUB-ai/releases)

---

**Status**: Stage 2 Complete ✅ | Models via Releases 📦 | Ready to Use 🚀

**Get Started**: `python download_models.py && python sub_ai.py` 💬
