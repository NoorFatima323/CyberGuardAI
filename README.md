# CyberGuard AI 🚨

**CyberGuard AI** is an AI-powered assistant designed to detect spam, fraud, and potentially malicious calls in real-time. Using speech-to-text transcription and audio analysis, it classifies calls and suggests safe responses to protect users from scams.

---

## Features

- **Live-call detection**: Analyze recorded calls for spam or fraud.
- **Emotion & audio analysis**: Detects tone, pitch, and tempo of the caller.
- **Smart classification**: Uses heuristics and optional AI backend for accurate labeling.
- **Safe response suggestions**: Provides short, safe phrases to handle suspicious calls.

---

## Technologies Used

- Python
- [Whisper](https://github.com/openai/whisper) for speech-to-text
- [Librosa](https://librosa.org/) for audio feature extraction
- [Gradio](https://gradio.app/) for web interface (Colab-ready)
- Optional: [Groq API](https://www.groq.com/) for advanced classification

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CyberGuard-AI.git
cd CyberGuard-AI

# Install required packages
pip install -r requirements.txt
