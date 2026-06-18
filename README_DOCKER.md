# 🚗 Safe Driving Assistant — Docker Integration Guide

Welcome to the premium Docker integration guide for your **Safe Driving Assistant**. This guide provides step-by-step instructions on how to package, configure, build, and run the assistant inside a fully isolated, headless-capable Docker container.

The containerized environment includes all C/C++ libraries, graphics backends, sound drivers, and speech synthesizers, pre-configured with a virtual framebuffer (**Xvfb**) to prevent OpenCV graphical crashes.

---

## 🌟 Highlights of the Docker Configuration

*   **Zero-Dependency Setup:** No need to install `dlib`, `cmake`, `opencv`, or `pyaudio` manually on your host machine.
*   **Virtual Framebuffer (Xvfb):** Headless-safe. If no display is detected, it automatically spawns a virtual X11 server so that `cv2.imshow` calls do not crash.
*   **Dynamic Network Routing:** Pre-configured to easily bridge to your local **Ollama** SLM instance.
*   **Unified Sound Backend:** Installs PulseAudio, ALSA, and `espeak` dependencies required for speech synthesis and playback.
*   **Environment-Driven Configuration:** Highly customizable via environment variables (`FLASK_HOST`, `CAMERA_ID`, `OLLAMA_API_URL`, etc.).

---

## 🛠️ Step 1: Build the Docker Image

Open your terminal in the project directory (`Safe Driving Assistant`) and run the following command to build your custom co-pilot image:

```bash
docker build -t safe-driving-assistant:latest .
```

*This compilation will take several minutes during the first run because it builds `dlib` (facial recognition) and compiles native C extensions.*

---

## 🚀 Step 2: Run the Docker Container

Depending on your host Operating System and hardware setup, select one of the premium run configurations below:

### Option A: Standard Headless / Web-HUD Only (Recommended)
This runs the assistant, serves the live futuristic Web-HUD on port `5000`, and bridges network queries to your host's local Ollama service.

```bash
docker run -d \
  --name driving_assistant \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_API_URL="http://host.docker.internal:11434/api/generate" \
  -p 5000:5000 \
  safe-driving-assistant:latest
```

*   **Web Dashboard:** Once started, open your browser and navigate to **`http://localhost:5000`** to view the live dashboard!
*   **How it works:** The container runs headlessly. Face recognition and eye-aspect ratio tracking are active, and the live stream is pushed directly to the dashboard.

---

### Option B: Linux Run with Full Hardware Passthrough (Camera & Audio)
If you are running native Linux and want the container to access your physical USB webcam and audio hardware directly, run:

```bash
docker run -it \
  --name driving_assistant \
  --device /dev/video0:/dev/video0 \
  --device /dev/snd:/dev/snd \
  --add-host=host.docker.internal:host-gateway \
  -e CAMERA_ID=0 \
  -e OLLAMA_API_URL="http://host.docker.internal:11434/api/generate" \
  -p 5000:5000 \
  safe-driving-assistant:latest
```

*   **`--device /dev/video0`:** Mounts your primary webcam inside the container.
*   **`--device /dev/snd`:** Exposes speaker and mic controls.

---

### Option C: Windows Host (WSL2) with Web USB Webcam Passthrough
If you are using WSL2 on Windows and want to feed your physical webcam into the Docker container:

1.  Bind your USB camera to WSL2 using [usbipd-win](https://github.com/dorssel/usbipd-win):
    ```powershell
    usbipd list
    usbipd bind --busid <BUSID>
    usbipd attach --wsl --busid <BUSID>
    ```
2.  Start the container with device mapping:
    ```bash
    docker run -it \
      --name driving_assistant \
      --device /dev/video0:/dev/video0 \
      --add-host=host.docker.internal:host-gateway \
      -e CAMERA_ID=0 \
      -e OLLAMA_API_URL="http://host.docker.internal:11434/api/generate" \
      -p 5000:5000 \
      safe-driving-assistant:latest
    ```

---

## ⚙️ Environment Variables Customization

You can dynamically tune your assistant container at runtime by passing `-e KEY=VALUE` parameters to `docker run`:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `FLASK_HOST` | `0.0.0.0` | Binding IP address for Flask web dashboard. |
| `FLASK_PORT` | `5000` | Port on which the web HUD dashboard will be served. |
| `CAMERA_ID` | `0` | Camera index. |
| `OLLAMA_API_URL` | `http://localhost:11434/api/generate` | The API endpoint for the Ollama voice co-pilot. Use `http://host.docker.internal:11434/api/generate` for bridging to host. |
| `OLLAMA_MODEL` | `drivesafe` | Name of the custom SLM model configured inside Ollama. |
| `FRAME_WIDTH` | `640` | Video frame capture width. |
| `FRAME_HEIGHT` | `480` | Video frame capture height. |

---

## 🧼 Housekeeping & Diagnostics

### View Real-Time Logs
To see the system warnings, detected EAR, and conversational chatbot interactions:
```bash
docker logs -f driving_assistant
```

### Stop & Remove Container
To stop and clean up the assistant container instance:
```bash
docker stop driving_assistant
docker rm driving_assistant
```

---
**Safe Driving Assistant** — *Keep your eyes on the road, your hands on the wheel, and drive safely!* 🚗💨
