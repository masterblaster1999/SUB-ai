# SUB ai - Small Language Model

## Overview
SUB ai is a comprehensive small language model with dual capabilities:
1. **Number Detection**: Detect and recognize digits (0-9) from images
2. **Conversational AI**: Natural text-based conversations like ChatGPT

## 🌟 Features

### Number Detection Module
- ✅ Detect numbers from images (digits 0-9)
- ✅ Classify images as "number" or "not a number"
- ✅ CNN-based neural network for high accuracy
- ✅ Trained on MNIST dataset (98-99% accuracy)

### Chat AI Module
- ✅ Text-based conversations
- ✅ Natural language understanding
- ✅ Sequence-to-sequence neural architecture
- ✅ Rule-based fallback for reliability
- ✅ Interactive chat interface

## 📦 Project Structure
```
SUB-ai/
├── README.md
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
4. Wait ~3-5 minutes
5. Model automatically commits to repository

### Option 2: Local Training

```bash
# Train number detection model
python train.py

# Train chat model
python train_chat.py
```

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
| Chat AI | Training Time | 3-5 min |
| Chat AI | Model Size | ~2 MB |
| Chat AI | Response Time | <50ms |

## 🛣️ Roadmap

### Stage 1 ✅ (Completed)
- [x] Number detection from images
- [x] CNN model training on MNIST
- [x] GitHub Actions workflow
- [x] 98%+ accuracy

### Stage 2 🚧 (In Progress)
- [x] Conversational AI chat
- [x] Text response generation
- [ ] Improve chat training data
- [ ] Add context memory

### Stage 3 (Next)
- [ ] Multi-digit number recognition
- [ ] Combine vision + language (VQA)
- [ ] Real-world image support
- [ ] Web interface

### Stage 4 (Future)
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Mobile app
- [ ] API endpoints

## 📚 Example Conversations

### General Chat
```
You: Hi there!
SUB ai: Hi there! I'm SUB ai, your AI assistant.

You: What is AI?
SUB ai: AI stands for Artificial Intelligence - computer systems that can learn and reason.
```

### About Capabilities
```
You: What can you do?
SUB ai: I can detect numbers from images and chat with you!

You: Tell me about yourself
SUB ai: I'm SUB ai, created to assist with number detection and conversations.
```

### Number Detection
```
You: detect test_images/number_7.png
SUB ai: This is a number image! Detected: 7
  Digit: 7
  Confidence: 99.23%
```

## 👥 Commands

When using `sub_ai.py`:

- **Chat**: Just type normally
- **Detect**: `detect <image_path>`
- **Help**: `help`
- **Exit**: `quit` or `exit`

## 🤖 GitHub Actions

Automated workflows available:

1. **Train SUB ai Model** - Trains number detection
2. **Train SUB ai Chat Model** - Trains conversation AI
3. **Test SUB ai Model** - Runs tests on changes

See [WORKFLOWS.md](WORKFLOWS.md) for detailed guide.

## 🛠️ Technologies

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **LSTM/GRU** - Sequence modeling
- **Matplotlib** - Visualization

## 🐛 Troubleshooting

### Models Not Found
```bash
# Train both models
python train.py
python train_chat.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Low Accuracy
- Increase training epochs
- Use more training data
- Adjust model architecture

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Larger conversation datasets
- [ ] Better chat responses
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Web UI

## 📝 License

MIT License

## ✍️ Author

**Subhobhai** - [@subhobhai943](https://github.com/subhobhai943)
- Portfolio: [subhadip-portofolio.netlify.app](https://subhadip-portofolio.netlify.app)
- Building AI projects and experimenting with ML

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow team
- Open source AI community

---

**Status**: Stage 2 In Progress 🚧 | Chat AI Added ✅ | Ready for Conversations 💬

**Try it now**: `python sub_ai.py` 🚀
