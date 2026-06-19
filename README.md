# 🚗 AI-SAFE-DRIVING-ASSISTANT

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Python](https://img.shields.io/badge/python-3.10-blue)

## Introduction

**AI-SAFE-DRIVING-ASSISTANT** is a browser-based driver safety copilot named **Lara**. It watches the driver through the device webcam, calculates eye-closure duration in real time, and escalates through a tiered alert system — nudge, warning, and a simulated emergency response — if drowsiness is detected. Alongside safety monitoring, Lara doubles as a hands-free voice assistant: drivers can ask road-related questions or request music by voice, without ever touching the screen.

The project exists to explore how lightweight, browser-deployable computer vision (no specialized hardware) combined with an LLM-backed voice interface can reduce distracted and drowsy driving.

## ✨ Features

- 😴 **Real-time drowsiness detection** — Computes Eye Aspect Ratio (EAR) from webcam frames using facial landmarks to detect prolonged eye closure.
- 🚦 **Tiered alert escalation** — Issues a wake-up nudge at 5 seconds of closed eyes, a serious warning at 8 seconds, and tracks repeated drowsy events.
- 🆘 **Simulated emergency response** — If the driver doesn't respond to a rest prompt within 15 seconds, the system escalates to a simulated emergency call action.
- 🗣️ **Voice copilot ("Lara")** — Wake-word activated (`"Lara"`) voice assistant that answers driving and road-related questions via an LLM (Llama 3.1 8B Instruct over the NVIDIA NIM API), and politely declines unrelated topics.
- 🎵 **Voice-controlled music** — Search and play songs via YouTube Music by voice, with play/pause/skip/volume/mute controls and a `yt-dlp` fallback if the music API is blocked.
- 🐳 **Docker-ready** — Ships with a `Dockerfile` configured for Gunicorn and Hugging Face Spaces deployment.

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **build-essential**, **cmake**, and **g++** (required to compile `dlib`/`face_recognition`)
- An **NVIDIA NIM API key** (used for the voice assistant's LLM responses; the app runs in offline mode without it)
- A webcam-enabled browser environment

### Installation

```bash
# Clone the repository
git clone https://github.com/themehmi/AI-SAFE-DRIVING-ASSISTANT.git
cd AI-SAFE-DRIVING-ASSISTANT

# (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` installs a custom `face_recognition` fork directly from GitHub, so an internet connection is required during installation.

### Configuration

Create a `.env` file in the project root with the following variable:

```env
API_KEY=your_nvidia_nim_api_key_here
```

- `API_KEY` — Used to authenticate against the NVIDIA NIM endpoint (`https://integrate.api.nvidia.com/v1`) for the `meta/llama-3.1-8b-instruct` model. If omitted, the voice assistant runs in **Offline Mode** and skips LLM responses.

## 💻 Usage

Run the app locally with Flask's built-in server:

```bash
python app.py
```

The app will be available at `http://localhost:7860`. Open it in a webcam-enabled browser, say **"Lara"** to activate the voice assistant, and let the page run in the background while driving to enable drowsiness monitoring.

For production, the included `Dockerfile` runs the app with Gunicorn instead:

```bash
docker build -t safe-driving-assistant .
docker run -p 7860:7860 --env-file .env safe-driving-assistant
```

## 📂 Project Structure

```
AI-SAFE-DRIVING-ASSISTANT/
├── templates/             # HTML front-end (camera feed, voice UI, music player)
├── app.py                 # Flask backend: drowsiness detection, voice API, LLM & music logic
├── requirements.txt       # Python dependencies (Flask, OpenCV, face_recognition, ytmusicapi, etc.)
├── Dockerfile             # Container build for Gunicorn / Hugging Face Spaces deployment
├── README.md              # Project documentation (this file)
└── README_DOCKER.md        # Docker-specific deployment notes
```

## 🤝 Contributing

Contributions are welcome! To get started:

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Make your changes and commit them with clear messages.
4. Push to your fork and open a **Pull Request** describing the change.

Please keep PRs focused on a single change, and include relevant testing notes (e.g., how drowsiness/voice behavior was verified) where applicable.

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details (add one if it isn't present yet).
