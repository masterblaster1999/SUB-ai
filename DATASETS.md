# SUB ai - Training Datasets Guide

## Overview

SUB ai can be trained on multiple datasets from Hugging Face for high-quality conversational capabilities. This guide explains the available datasets and how to use them.

## Available Datasets

### 1. DailyDialog (Default) 🌟

**Dataset**: `daily_dialog`
**Source**: [Hugging Face - DailyDialog](https://huggingface.co/datasets/daily_dialog)

#### Description
- High-quality multi-turn dialogues about daily life
- 13,118 dialogues with 8 conversation topics
- Natural, human-like conversations
- Perfect for general-purpose chatbots

#### Topics Covered
- Ordinary life
- School life  
- Culture & education
- Attitude & emotion
- Relationship
- Tourism
- Health
- Work

#### Statistics
- **Total Dialogues**: ~13,000
- **Average Turns per Dialogue**: 8
- **Vocabulary Size**: ~13,000 words
- **Language**: English

#### Sample Conversation
```
Person A: "Good morning! How are you today?"
Person B: "I'm doing great, thanks! How about you?"
Person A: "Pretty good! Do you have any plans for the weekend?"
Person B: "Yes, I'm planning to visit the museum."
```

#### When to Use
- ✅ General conversation AI
- ✅ Daily life discussions
- ✅ Polite and natural interactions
- ✅ Multi-turn dialogues

---

### 2. Empathetic Dialogues 💖

**Dataset**: `empathetic_dialogues`
**Source**: [Hugging Face - Empathetic Dialogues](https://huggingface.co/datasets/empathetic_dialogues)

#### Description
- Conversations grounded in emotional situations
- 25,000+ conversations with empathetic responses
- Focuses on understanding and responding to emotions
- Great for more human-like, caring AI

#### Emotion Categories
- Happy, Sad, Angry, Surprised
- Afraid, Disgusted, Anxious, Excited
- And 24 more emotion categories

#### Statistics
- **Total Conversations**: ~25,000
- **Emotion Categories**: 32
- **Average Length**: 4.3 turns
- **Language**: English

#### Sample Conversation
```
Person A: "I'm feeling really anxious about my exam tomorrow."
Person B: "I understand how stressful that can be. Have you prepared well?"
Person A: "Yes, but I still feel nervous."
Person B: "That's completely normal! Just do your best and you'll be fine."
```

#### When to Use
- ✅ Empathetic AI assistants
- ✅ Emotional support chatbots
- ✅ More human-like responses
- ✅ Understanding user emotions

---

### 3. Local Dataset (Fallback)

**Dataset**: Built-in local conversations
**Source**: `train_chat.py` (hardcoded)

#### Description
- 60+ base conversation pairs
- 130+ with variations (capitalization)
- Focused on SUB ai's capabilities
- No internet required

#### Topics Covered
- Greetings and introductions
- SUB ai capabilities
- AI/ML concepts
- Numbers and detection
- General chat

#### Statistics
- **Total Pairs**: 130+
- **Topics**: ~15
- **Language**: English
- **Size**: Very small

#### Sample Conversation
```
User: "What can you do?"
SUB ai: "I can detect numbers from images and chat with you!"
```

#### When to Use
- ✅ Testing without internet
- ✅ Quick local training
- ✅ Minimal setup required
- ✅ Learning SUB ai basics

---

## Training Parameters

### Recommended Settings

| Dataset | Max Samples | Epochs | Batch Size | Training Time |
|---------|-------------|--------|------------|---------------|
| DailyDialog | 5,000 | 100 | 32 | 8-12 min |
| Empathetic | 5,000 | 100 | 32 | 8-12 min |
| Local | All (~130) | 100 | 32 | 3-5 min |

### For Better Results

**More Data**:
```bash
export MAX_SAMPLES=10000  # Use 10K samples
python train_chat.py
```

**More Training**:
```bash
export CHAT_EPOCHS=150  # Train longer
python train_chat.py
```

**Larger Batch**:
```bash
export CHAT_BATCH_SIZE=64  # Faster but needs more memory
python train_chat.py
```

---

## Training on GitHub Actions

### Step-by-Step

1. **Navigate to Actions**
   - Go to your repository
   - Click the "Actions" tab

2. **Select Workflow**
   - Click "Train SUB ai Chat Model"
   - Click "Run workflow" button

3. **Configure Parameters**
   - **Epochs**: 100 (default)
   - **Batch Size**: 32 (default)
   - **Dataset**: Choose from:
     - `daily_dialog` (recommended)
     - `empathetic_dialogues`
     - `local` (fallback)
   - **Max Samples**: 5000 (default)

4. **Start Training**
   - Click "Run workflow"
   - Wait 8-12 minutes
   - Model auto-commits to repository

---

## Local Training

### With DailyDialog

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings
python train_chat.py
```

### With Empathetic Dialogues

```bash
# Set environment variable
export HF_DATASET=empathetic_dialogues
export MAX_SAMPLES=5000

# Train
python train_chat.py
```

### With Local Data Only

```bash
# Disable Hugging Face datasets
export USE_HF_DATASET=false

# Train
python train_chat.py
```

---

## Performance Comparison

### Dataset Quality Impact

| Metric | Local | DailyDialog | Empathetic |
|--------|-------|-------------|------------|
| Response Variety | Low | High | Very High |
| Naturalness | Medium | High | Very High |
| Context Understanding | Low | High | High |
| Emotion Recognition | Low | Medium | High |
| Training Time | Fast | Medium | Medium |

### Expected Results

**After Local Training**:
```
User: "Hello!"
SUB ai: "Hello! How can I help you today?"

User: "What's the weather?"
SUB ai: "That's interesting! Tell me more." (generic)
```

**After DailyDialog Training**:
```
User: "Hello!"
SUB ai: "Hi there! How are you doing today?"

User: "What's the weather?"
SUB ai: "I don't have real-time weather data, but I hope it's nice where you are!"
```

**After Empathetic Training**:
```
User: "I'm feeling stressed."
SUB ai: "I understand that can be difficult. Is there anything specific bothering you?"

User: "Work is overwhelming."
SUB ai: "That sounds really tough. Remember to take breaks when you need them!"
```

---

## Troubleshooting

### Dataset Download Issues

**Error**: `ConnectionError` or dataset not found

**Solution**:
```bash
# Use fallback local data
export USE_HF_DATASET=false
python train_chat.py
```

### Memory Issues

**Error**: `OOM` (Out of Memory)

**Solution**:
```bash
# Reduce samples
export MAX_SAMPLES=1000
# Or reduce batch size
export CHAT_BATCH_SIZE=16
python train_chat.py
```

### Slow Training

**Issue**: Training takes too long

**Solution**:
```bash
# Reduce samples or epochs
export MAX_SAMPLES=2000
export CHAT_EPOCHS=50
python train_chat.py
```

---

## Future Datasets

Potential datasets to add:

- ☐ **OpenAssistant** - Instruction-following conversations
- ☐ **Persona-Chat** - Personality-based dialogues  
- ☐ **BlendedSkillTalk** - Multi-skill conversations
- ☐ **Multi-WOZ** - Task-oriented dialogues
- ☐ **Custom Domain** - Specialized topic conversations

---

## Best Practices

### For General Chat
1. Use **DailyDialog** dataset
2. Train with **5,000+ samples**
3. Use **100 epochs**
4. Test with diverse questions

### For Empathetic AI
1. Use **Empathetic Dialogues** dataset
2. Train with **5,000+ samples**
3. Consider **150 epochs** for better emotion understanding
4. Test with emotional scenarios

### For Quick Testing
1. Use **local** dataset
2. Train with **all samples** (~130)
3. Use **50-100 epochs**
4. Good for development and testing

---

## Contributing

Want to add more datasets? Contributions welcome!

1. Fork the repository
2. Add dataset loading code to `train_chat.py`
3. Update this documentation
4. Submit a pull request

---

**Happy Training! 🚀**

For questions, open an [issue](https://github.com/subhobhai943/SUB-ai/issues).
