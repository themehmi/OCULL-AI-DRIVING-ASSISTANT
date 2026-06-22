# 🚗 OCULL-AI-SAFE-DRIVING-ASSISTANT

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-yellow)
![Python](https://img.shields.io/badge/python-3.10-blue)

## Introduction

**AI-SAFE-DRIVING-ASSISTANT** is a browser-based driver safety copilot named **Lara**. It watches the driver through the device webcam, calculates eye-closure duration in real time, and escalates through a tiered alert system — nudge, warning, and a simulated emergency response — if drowsiness is detected. Alongside safety monitoring, Lara doubles as a hands-free voice assistant: drivers can ask road-related questions or request music by voice, without ever touching the screen.

The project exists to explore how lightweight, browser-deployable computer vision (no specialized hardware) combined with an LLM-backed voice interface can reduce distracted and drowsy driving.

## 🌐 Live Demo

A hosted version is running on Hugging Face Spaces:

**👉 [https://huggingface.co/spaces/themehmi/AI-Safe-Driving-Assistant](https://huggingface.co/spaces/themehmi/AI-Safe-Driving-Assistant)**

Open it in a browser with webcam and microphone access enabled to try the drowsiness monitor and voice copilot directly — no installation required.

## 🧠 How It Works

The system is a single Flask app that ties together three loosely-coupled loops running in the browser: a **vision loop**, a **voice loop**, and a **status loop**. Here's how data flows between them:

```
┌─────────────────────┐        webcam frame (base64)        ┌──────────────────────┐
│   Browser (index)    │ ───────────────────────────────────▶ │  POST /api/frame      │
│  - captures webcam   │                                      │  - decode + cv2       │
│  - captures mic via  │                                      │  - face_recognition   │
│    Web Speech API    │ ◀─────────────────────────────────── │  - compute EAR        │
└─────────────────────┘     state, EAR, alert, action          └──────────────────────┘
        │      ▲
        │      │ transcribed speech text                Voice command / question
        ▼      │
┌─────────────────────┐                                  ┌──────────────────────┐
│  POST /api/voice     │ ───────────────────────────────▶ │ Wake-word + dialogue   │
│                       │                                  │ state machine          │
└─────────────────────┘ ◀─────────────────────────────────│ LLM (Llama 3.1 8B via  │
                              spoken reply / music action   │ NVIDIA NIM) for Q&A    │
                                                            │ ytmusicapi / yt-dlp     │
                                                            │ for music search        │
                                                            └──────────────────────┘
```

**1. Drowsiness monitoring (`/api/frame`)**
The front end repeatedly grabs a webcam frame, base64-encodes it, and POSTs it to the backend. On the server, OpenCV decodes the image and `face_recognition` extracts eye landmarks. From the six landmark points per eye, the backend computes the **Eye Aspect Ratio (EAR)** — a ratio of vertical-to-horizontal eye distances that drops sharply when eyes close. The EAR is compared against a fixed threshold (`0.22`):

- **Eyes stay closed ≥ 5 seconds** → state moves to `LEVEL_1`, and the app speaks a wake-up nudge.
- **Eyes stay closed ≥ 8 seconds** → state escalates to `LEVEL_3`, the drowsy-event counter increments, and a stronger warning is issued.
- **3+ drowsy events recorded** → the assistant proactively asks via voice whether the driver wants to pull over and rest.
- **No response to that prompt within 15 seconds** → state escalates to `CRITICAL` and the backend signals a simulated emergency-call action to the front end.

This all happens statelessly per-request on the server, with the running counters (`system_status`, dialogue state, timers) held in global memory between calls.

**2. Voice copilot ("Lara") (`/api/voice`)**
The browser's speech recognition transcribes spoken audio to text and POSTs it to `/api/voice`. The backend implements a small state machine:
- If a **drowsiness dialogue** is active (e.g., "do you want to rest?" or "what song should I play?"), incoming speech is interpreted as the answer to that specific question.
- Otherwise, the backend listens for the wake word **"Lara"**. Once activated, simple commands (pause, resume, skip, volume, mute) are handled directly; anything else is forwarded as a prompt to an LLM (`meta/llama-3.1-8b-instruct`, served via the NVIDIA NIM API) with a system prompt that restricts it to driving- and road-related answers only.

**3. Voice-controlled music**
Song requests (either from a direct "play [song]" command or from the drowsiness dialogue's "play a song to stay awake" branch) are resolved by searching YouTube Music via `ytmusicapi`. If that's blocked or fails (a common issue on cloud hosts), the backend falls back to a `yt-dlp` search. Returned video IDs are sent back to the browser, which plays the audio natively.

**4. Status polling (`/api/status`)**
A lightweight endpoint the front end can poll to read the current state (`NORMAL` / `LEVEL_1` / `LEVEL_3` / `CRITICAL`), EAR value, and drowsy-event counter — useful for driving any on-screen indicators (e.g., the eye-tracking overlay).

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
